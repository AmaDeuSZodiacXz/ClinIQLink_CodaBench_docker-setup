from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import torch
import faiss
import pickle
import argparse

class DPR:
    def __init__(self, moodel_name = "BMRetriever/BMRetriever-410M"):
        self.model = AutoModel.from_pretrained(moodel_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(moodel_name)
        self.index = None
        self.metadata = None
        
    def last_token_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            embedding = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
        return embedding

    def get_detailed_instruct_query(self, task_description: str, query: str) -> str:
        return f'{task_description}\nQuery: {query}'

    def get_detailed_instruct_passage(self, passage: str) -> str:
        return f'Represent this passage\npassage: {passage}'

    def embed(self, docs, batch_size=32, max_length = 1024, verbose=True, query = False):
        """
        Embeds a list of documents using the model in batches with memory optimizations.

        This function processes the documents in batches, moving each batch's
        embeddings to CPU immediately after computation, and clears the CUDA cache
        after each batch to help with GPU memory management. It also uses tqdm to show
        progress during both tokenization and indexing.

        Args:
            docs (List[str]): List of input documents.
            batch_size (int): Number of documents to process per batch.
            verbose (bool): Whether to display progress bars.

        Returns:
            torch.Tensor: A tensor of shape (len(docs), hidden_size) containing embeddings.
        """
        # Preprocess documents with a progress bar
        if not query:
          documents = [self.get_detailed_instruct_passage(doc) for doc in tqdm(docs, desc="Preprocessing Documents", disable=not verbose)]
        else:
          documents = docs

        # Tokenize all documents at once
        batch_dict = self.tokenizer(
            documents,
            max_length=max_length - 1,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Append EOS token to each input_ids
        input_ids_with_eos = [ids.tolist() + [self.tokenizer.eos_token_id] for ids in batch_dict['input_ids']]

        # Re-pad the input_ids and regenerate the attention mask
        full_batch = self.tokenizer.pad(
            {'input_ids': input_ids_with_eos},
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move tensors to GPU
        full_batch = {k: v.to(self.device) for k, v in full_batch.items()}

        total = full_batch['input_ids'].size(0)
        embeddings_list = []

        self.model.eval()
        with torch.no_grad():
            # Process documents in batches with progress bar
            for i in tqdm(range(0, total, batch_size), disable=not verbose, desc="Indexing Batches"):
                # Create a mini-batch
                batch = {k: v[i:i+batch_size] for k, v in full_batch.items()}

                # Forward pass through the model
                outputs = self.model(**batch)

                # Pool embeddings using your custom function
                batch_embeds = self.last_token_pool(outputs.last_hidden_state, batch['attention_mask'])

                # Move embeddings to CPU immediately to free GPU memory
                batch_embeds = batch_embeds.detach().cpu()
                embeddings_list.append(batch_embeds)

                # Clear CUDA cache to help with memory fragmentation
                torch.cuda.empty_cache()

        # Concatenate all batch embeddings into one tensor
        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings
    
    def retrieve(self, query, top_k=10, verbose=0):
        task = 'Given a scientific claim, retrieve documents that support or refute the claim'
        queries = [
            self.get_detailed_instruct_query(task, query),
        ]
        query_vec = self.embed(queries, query = True)
        distances, indices = self.index.search(query_vec.cpu(), top_k)
        result = [self.metadata[idx]['chunk'] for idx in indices[0]]
        if verbose:
            print("Query:", query)
            print("\nTop Results:")
            for i, idx in enumerate(indices[0]):
                print(f"{i+1}. {self.metadata[idx]['chunk']} (Distance: {distances[0][i]:.4f})")

        return result, distances
        
    def add_index(self, embedding, new_metadata):
        self.index.add(embedding.cpu())

        self.metadata.extend(new_metadata)

    def load_index(self, index_path="faiss_index.bin", metadata_path="metadata.pkl"):
        """
        Loads the FAISS index and metadata from disk.

        Args:
            index_path (str): File path to load the FAISS index from.
            metadata_path (str): File path to load the metadata from.

        Returns:
            index (faiss.Index): The loaded FAISS index.
            metadata (list): The loaded metadata.
        """
        # Load FAISS index from the binary file
        self.index = faiss.read_index(index_path)
        print(f"FAISS index loaded from {index_path}")

        # Load metadata using pickle
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Metadata loaded from {metadata_path}")

    def get_index(self):
        return self.index, self.metadata

class Reranker:
    def __init__(self, model_name = "BMRetriever/BMRetriever-2B"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def last_token_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            embedding = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
        return embedding

    def get_detailed_instruct_query(self, task_description: str, query: str) -> str:
        return f'{task_description}\nQuery: {query}'

    def get_detailed_instruct_passage(self, passage: str) -> str:
        return f'Represent this passage\npassage: {passage}'
    
    def embed(self, docs: list, query = False, max_length = 512):
        """
        docs: list of documents're strings
        query: use to indicate if it is query or normal passage
        """

        if query:
            # Each query must come with a one-sentence instruction that describes the task
            task = 'Given a scientific claim, retrieve documents that support or refute the claim'
            input_texts = [
                self.get_detailed_instruct_query(task, query)
            ]
        else:
            # No need to add instruction for retrieval documents
            input_texts = [
                self.get_detailed_instruct_passage(i) for i in docs
            ]

        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=max_length-1, padding=True, truncation=True, return_tensors='pt')

        input_ids_with_eos = [ids.tolist() + [self.tokenizer.eos_token_id] for ids in batch_dict['input_ids']]
        # Re-pad the input_ids (and regenerate the attention mask) from lists.
        padded = self.tokenizer.pad(
            {'input_ids': input_ids_with_eos},
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move all tensors in padded to GPU
        for key in padded:
            padded[key] = padded[key].to("cuda")
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**padded)
            embeddings = self.last_token_pool(outputs.last_hidden_state, padded['attention_mask'])
        return embeddings
    
    def sort_and_map(self, input_list):
        """
        Sorts the input list in descending order and creates a mapping from the original index
        (0-based) to the new index in the sorted list (0-based).

        Parameters:
            input_list (list): A list of comparable and unique elements.

        Returns:
            tuple: A tuple containing:
                - sorted_list (list): The list sorted in descending order.
                - index_mapping (dict): A dictionary mapping each element's original index to its new index.
        """
        # Sort the list in descending order
        sorted_list = sorted(input_list, reverse=True)

        # Build a dictionary mapping each unique number to its new index in the sorted list.
        # This works efficiently when all elements are unique.
        new_index_map = {num: i for i, num in enumerate(sorted_list)}

        # Create a mapping from the original index to the new index.
        index_mapping = {orig_index: new_index_map[num] for orig_index, num in enumerate(input_list)}

        return sorted_list, index_mapping

    def ranking(self, query, docs):
        """_summary_

        Args:
            query (_type_): _description_
            docs (_type_): _description_

        Returns:
            list: list of reranked retrieved list
            score: score of each entry
        """
        query_vec = self.embed(query, True)
        doc_vec = self.embed(docs)
        score = query_vec @ doc_vec.T
        # s = torch.nn.Softmax(dim=-1)
        sorted_list, idx = self.sort_and_map(score.cpu().tolist()[0])

        print(sorted_list, idx)
        return [docs[i] for i in list(idx.values())] , sorted_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      prog='top',
      description='Show top lines from each file')
    parser.add_argument('-q', '--query')
    args = parser.parse_args()
    # gen_query = "How do the intrinsic muscles of the hand differ from the extrinsic muscles, and why are the intrinsic muscles vital for functions like pinch and grip strength? "
    gen_query = args.query
    dpr = DPR()
    dpr.load_index(index_path="./index/faiss_index.bin", metadata_path="./index/statpearls_meta.pkl")
    search_result, score = dpr.retrieve(gen_query, verbose = 1, top_k=30)

    ranker = Reranker("BMRetriever/BMRetriever-2B")
    ranked_search_result, rank_score = ranker.ranking(gen_query, search_result)

    print("-"*15,"reranked","-"*15)
    for i, r in enumerate(zip(ranked_search_result, rank_score)):
        d,s = r
        print(f"{i}. {d} \n score: {s}")