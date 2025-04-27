# --- Start: submit.py (Prediction Only - Compliant with Docker Setup) ---
import json
import os
import random
import argparse
import torch
# Keep transformers imports needed for loading models/pipelines
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline, GenerationConfig, BitsAndBytesConfig
from transformers.pipelines import TextGenerationPipeline
import re
# Imports needed for DPR/Reranker and RAG logic
import numpy as np
import faiss # Requires faiss-cpu or faiss-gpu to be installed
import pickle
import traceback
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/transformers"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# --- Import User's Custom DPR and Reranker Classes ---
try:
    from dpr import DPR, Reranker # Assumes dpr.py is in /app
    print("Successfully imported custom DPR and Reranker classes from dpr.py.", flush=True)
except ImportError as e:
    print(f"ERROR: Could not import custom DPR or Reranker classes from dpr.py. {e}", flush=True)
    import sys; sys.exit(1)
# --- End Custom Class Import ---


# --- Constants for Model Paths ---
# Base directory for models within the container (mounted or copied)
# Assumes the Dockerfile/Singularity build process makes models available here
MODEL_BASE_PATH = "../model_submission"

# Construct model paths relative to MODEL_BASE_PATH
PATH_BMRETRIEVER_410M = os.path.join(MODEL_BASE_PATH, "BMRetriever-410M")
PATH_BMRETRIEVER_2B = os.path.join(MODEL_BASE_PATH, "BMRetriever-2B")
PATH_GENERATOR_LLM = os.path.join(MODEL_BASE_PATH, "Qwen2.5-7B-Instruct")
PATH_VERIFIER_LLM = os.path.join(MODEL_BASE_PATH, "MedReason-8B")

# Index paths (Keep as absolute paths, assuming mounted access on HPC/container)
PATH_VECTOR_INDEX = os.path.join(MODEL_BASE_PATH, "stat_pubmed_indexB1.bin")
PATH_METADATA = os.path.join(MODEL_BASE_PATH, "stat_pubmed_indexB1.pkl")

# --- Configuration ---
compute_dtype = torch.bfloat16
DEFAULT_MAX_LEN_GENERATOR = 32768
DEFAULT_MAX_LEN_VERIFIER = 8192
RESERVED_TOKENS_LLM = 1024

class ClinIQLinkSampleDatasetSubmit:
    # --- Modified __init__ to load ALL necessary components for inference ---
    def __init__(self, run_mode="container", max_length=1028, sample_sizes=None, random_sample=False, chunk_size=2,
                    do_sample=False, temperature=None, top_p=None, top_k=None):

        self.DEBUG = True # Keep debug prints active
        print(f"--- Debug Mode: {'ON' if self.DEBUG else 'OFF'} ---", flush=True)

        self.run_mode = run_mode.lower()
        self.output_max_length = max_length # Store arg for generation max_new_tokens if needed
        self.sample_sizes = sample_sizes or {}
        self.random_sample = random_sample
        self.chunk_size = chunk_size # Batch size for inference

        # Generation config from args
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.do_sample and (self.temperature is None or self.temperature <= 0):
            print("[WARN] --do_sample set but temperature <=0; setting temp=0.7", flush=True)
            self.temperature = 0.7

        # Base directories
        if run_mode == "container":
            print("Running in container mode.", flush=True)
            self.base_dir = "/app" # Script runs from /app in container
        else: # local mode
            print("Running in local mode.", flush=True)
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Use ENV VAR for DATA_DIR ---
        self.dataset_dir = os.getenv("DATA_DIR", os.path.join(self.base_dir, "../data")) # Default for local
        self.template_dir = os.path.join(self.base_dir, "submission_template")
        print(f"Using dataset directory: {self.dataset_dir}", flush=True)
        print(f"Using template directory: {self.template_dir}", flush=True)
        # --- End ENV VAR ---

        # No NLTK setup or SentenceTransformer loading needed in submit.py

        # Original Placeholder calls (MUST be present, even if returning None)
        self.model = self.load_participant_model()
        self.pipeline = self.load_participant_pipeline()

        # --- Initialize Your Custom Components ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing custom components on device: {self.device}", flush=True)
        try: # DPR
            print("Initializing DPR Retriever...", flush=True)
            self.dpr_retriever = DPR(moodel_name=PATH_BMRETRIEVER_410M, device=self.device)
            print(f"Loading DPR index from: {PATH_VECTOR_INDEX} and metadata: {PATH_METADATA}", flush=True)
            if not os.path.exists(PATH_VECTOR_INDEX): raise FileNotFoundError(f"DPR Index not found: {PATH_VECTOR_INDEX}")
            if not os.path.exists(PATH_METADATA): raise FileNotFoundError(f"DPR Metadata not found: {PATH_METADATA}")
            self.dpr_retriever.load_index(index_path=PATH_VECTOR_INDEX, metadata_path=PATH_METADATA)
            print("DPR Retriever initialized.", flush=True)
        except Exception as e: print(f"FATAL ERROR initializing DPR: {e}", flush=True); raise e
        try: # Reranker
            print("Initializing Reranker...", flush=True)
            self.reranker = Reranker(model_name=PATH_BMRETRIEVER_2B)
            print("Reranker initialized.", flush=True)
        except Exception as e: print(f"FATAL ERROR initializing Reranker: {e}", flush=True); raise e

        self.participant_models = self._load_llm_models_custom()
        self._set_llm_max_lengths()
        self.participant_pipelines = self._load_participant_llm_pipelines_custom()
        # --- End Custom Component Loading ---

        # Load and sample the dataset (as per original template)
        self.sampled_qa_pairs = self.load_and_sample_dataset()
        self.SYSTEM_MSG = ( "You are a highly knowledgeable medical expert. Reply **only** with the requested answer format. Do not repeat the question or add explanations." )

    # --- Original Placeholder Methods (Keep as is) ---
    def load_participant_model(self):
        """ Original Placeholder - Participant should modify IF NOT loading models in __init__ """
        print("Original load_participant_model called (Placeholder - Models loaded in __init__)", flush=True)
        return None # Keep as None if models loaded elsewhere

    def load_participant_pipeline(self):
        """ Original Placeholder - Participant should modify IF NOT loading pipelines in __init__ """
        print("Original load_participant_pipeline called (Placeholder - Pipelines loaded in __init__)", flush=True)
        return None # Keep as None if pipelines loaded elsewhere
    # --- End Original Placeholders ---

    # --- Start: Your Custom Model/Pipeline Loading (Internal Use) ---
    def _load_llm_models_custom(self):
        print("Loading custom LLMs (Qwen2.5-7B, MedReason-8B)...", flush=True); models = {}
        try:
            print(f"Loading Generator from: {PATH_GENERATOR_LLM}", flush=True);
            if not os.path.isdir(PATH_GENERATOR_LLM): raise FileNotFoundError(f"Generator dir not found: {PATH_GENERATOR_LLM}")
            models['generator_tokenizer'] = AutoTokenizer.from_pretrained(PATH_GENERATOR_LLM, trust_remote_code=True); models['generator_model'] = AutoModelForCausalLM.from_pretrained(PATH_GENERATOR_LLM, torch_dtype=compute_dtype, trust_remote_code=True, device_map="auto").eval();
            if models['generator_tokenizer'].pad_token is None:
                if models['generator_tokenizer'].eos_token is not None: models['generator_tokenizer'].pad_token = models['generator_tokenizer'].eos_token; print("Gen tokenizer pad_token set eos.")
                else: models['generator_tokenizer'].add_special_tokens({'pad_token': '[PAD]'}); models['generator_model'].resize_token_embeddings(len(models['generator_tokenizer'])); print("Gen tokenizer added PAD token.")
            print(f"Loading Verifier from: {PATH_VERIFIER_LLM}", flush=True);
            if not os.path.isdir(PATH_VERIFIER_LLM): raise FileNotFoundError(f"Verifier dir not found: {PATH_VERIFIER_LLM}")
            models['verifier_tokenizer'] = AutoTokenizer.from_pretrained(PATH_VERIFIER_LLM, trust_remote_code=True); models['verifier_model'] = AutoModelForCausalLM.from_pretrained(PATH_VERIFIER_LLM, torch_dtype=compute_dtype, trust_remote_code=True, device_map="auto").eval();
            if models['verifier_tokenizer'].pad_token is None:
                 if models['verifier_tokenizer'].eos_token is not None: models['verifier_tokenizer'].pad_token = models['verifier_tokenizer'].eos_token; print("Verifier tokenizer pad_token set eos.")
                 else: models['verifier_tokenizer'].add_special_tokens({'pad_token': '[PAD]'}); models['verifier_model'].resize_token_embeddings(len(models['verifier_tokenizer'])); print("Verifier tokenizer added PAD token.")
            print("Custom LLMs loaded.", flush=True); return models
        except Exception as e: print(f"Error loading LLMs: {e}", flush=True); raise e

    def _set_llm_max_lengths(self):
        # (Keep implementation - unchanged)
        print("Setting custom LLM max lengths...", flush=True)
        try: # Generator
             config = self.participant_models['generator_model'].config
             if hasattr(config, 'sliding_window') and config.sliding_window: self.max_len_generator = config.sliding_window; print(f"Generator using sliding_window: {self.max_len_generator}")
             elif hasattr(config, 'max_position_embeddings'): self.max_len_generator = config.max_position_embeddings
             elif hasattr(config, 'seq_length'): self.max_len_generator = config.seq_length
             else: raise AttributeError("No known max length attr.")
        except AttributeError: self.max_len_generator = DEFAULT_MAX_LEN_GENERATOR; print(f"Warn: Gen default max_len: {self.max_len_generator}")
        try: # Verifier
             config = self.participant_models['verifier_model'].config
             if hasattr(config, 'max_position_embeddings'): self.max_len_verifier = config.max_position_embeddings
             elif hasattr(config, 'sliding_window') and config.sliding_window: self.max_len_verifier = config.sliding_window; print(f"Verifier using sliding_window: {self.max_len_verifier}")
             elif hasattr(config, 'seq_length'): self.max_len_verifier = config.seq_length
             else: raise AttributeError("No known max length attr.")
        except AttributeError: self.max_len_verifier = DEFAULT_MAX_LEN_VERIFIER; print(f"Warn: Verifier default max_len: {self.max_len_verifier}")
        print(f"Custom Max Lengths: Gen={self.max_len_generator}, Ver={self.max_len_verifier}", flush=True)

    def _load_participant_llm_pipelines_custom(self):
        # (Keep implementation - unchanged)
        print("Loading custom LLM pipelines...", flush=True); pipelines = {}
        if not self.participant_models: print("Models failed load.", flush=True); return None
        try:
            pipelines['generator_llm'] = pipeline( "text-generation", model=self.participant_models['generator_model'], tokenizer=self.participant_models['generator_tokenizer'], torch_dtype=compute_dtype, device_map="auto" )
            print("Qwen2.5 generator pipeline created.", flush=True)
            pipelines['qwen_verifier'] = pipeline( "text-generation", model=self.participant_models['verifier_model'], tokenizer=self.participant_models['verifier_tokenizer'], torch_dtype=compute_dtype, device_map="auto" )
            print("MedReason verifier pipeline created.", flush=True)
            print("Custom LLM pipelines loaded.", flush=True); return pipelines
        except Exception as e: print(f"Error loading custom LLM pipelines: {e}", flush=True); raise e
    # --- End: Your Custom Model/Pipeline Loading ---


    # --- Start: Original Framework Methods (Keep as is) ---
    def load_json(self, filepath): # Original
        try:
            with open(filepath, "r") as f: return json.load(f)
        except Exception as e: print(f"Error loading JSON from {filepath}: {e}", flush=True); return None

    def load_template(self, filename): # Original
        filepath = os.path.join(self.template_dir, filename)
        try:
            with open(filepath, "r") as f: return f.read()
        except Exception as e: print(f"Error loading template {filename} from {filepath}: {e}", flush=True); return None

    def load_and_sample_dataset(self): # Original
        qa_types = { "multiple_choice": ("MC.json", self.sample_sizes.get("num_mc", 200)), "true_false": ("TF.json", self.sample_sizes.get("num_tf", 200)), "list": ("list.json", self.sample_sizes.get("num_list", 200)), "short": ("short.json", self.sample_sizes.get("num_short", 200)), "short_inverse": ("short_inverse.json", self.sample_sizes.get("num_short_inv", 200)), "multi_hop": ("multi_hop.json", self.sample_sizes.get("num_multi", 200)), "multi_hop_inverse": ("multi_hop_inverse.json", self.sample_sizes.get("num_multi_inv", 200)), }
        sampled_qa = {}
        print(f"Loading datasets from: {self.qa_dir}", flush=True) # Use resolved path
        for qa_type, (filename, sample_size) in qa_types.items():
            filepath = os.path.join(self.qa_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f: data = json.load(f)
                flat_data = [item for sublist in data for item in (sublist if isinstance(sublist, list) else [sublist])]
                if self.random_sample: sampled_data = random.sample(flat_data, min(sample_size, len(flat_data)))
                else: sampled_data = flat_data[:sample_size]
                sampled_qa[qa_type] = sampled_data
            except FileNotFoundError: print(f"Warning: Dataset file not found: {filepath}", flush=True); sampled_qa[qa_type] = []
            except Exception as e: print(f"Error loading {filename}: {e}", flush=True); sampled_qa[qa_type] = []
        print(f"Successfully sampled {sum(len(v) for v in sampled_qa.values())} QA pairs.", flush=True); return sampled_qa

    def generate_prompt(self, template, qa, qa_type): # Original
        try:
            question = qa.get("question", "Unknown Question"); answer = qa.get("answer", "")
            options = qa.get("options", {}); reasoning = qa.get("reasoning", "")
            false_answer = qa.get("false_answer", "")
            # Original template doesn't use these, but keep for potential internal use if needed
            # incorrect_explanation=qa.get("incorrect_explanation", ""); incorrect_reasoning_step=qa.get("incorrect_reasoning_step", "")
            if qa_type == "true_false": return template.format(question=question)
            elif qa_type == "multiple_choice":
                 # Original expects options dict, convert if list
                 if not isinstance(options, dict): options = {chr(65 + i): opt for i, opt in enumerate(options)}
                 return template.format( question=question, options_A=options.get("A", "NA"), options_B=options.get("B", "NA"), options_C=options.get("C", "NA"), options_D=options.get("D", "NA") )
            elif qa_type == "list":
                options_list = qa.get("options", []) # Original template expected list of strings
                options_joined = "\n".join(options_list) if isinstance(options_list, list) else str(options_list)
                return template.format( question=question, options_joined=options_joined )
            elif qa_type == "multi_hop": return template.format(question=question)
            elif qa_type == "multi_hop_inverse":
                reasoning_str = "\n".join(reasoning) if isinstance(reasoning, list) else str(reasoning); return template.format( question=question, answer=answer, reasoning=reasoning_str )
            elif qa_type == "short": return template.format(question=question)
            elif qa_type == "short_inverse": return template.format( question=question, false_answer=false_answer )
            else: print(f"Warning: Unknown QA type '{qa_type}'", flush=True); return "Invalid QA type."
        except Exception as e: print(f"Error generating prompt: {e}", flush=True); return "Error generating prompt."

    def _strip_noise(self, text: str) -> str: # Original
        return re.sub(r"\bassistant\b", "", text, flags=re.I).strip()

    def _batched_inference(self, prompts, qa_type): # Original
        responses = []
        for i in range(0, len(prompts), self.chunk_size):
            chunk = prompts[i : i + self.chunk_size]
            # Calls participant_model (which calls orchestrator)
            out = self.participant_model(chunk if len(chunk) > 1 else chunk[0], qa_type=qa_type)
            if isinstance(out, list): responses.extend(out)
            else: responses.append(out)
        return responses

    def _bundle(self, inputs, responses, prompts=None): # Original
        bundled_inputs = []
        for i, qa in enumerate(inputs):
            item = qa.copy(); item["response"] = responses[i] # Store raw response
            if prompts: item["prompt"] = prompts[i]
            bundled_inputs.append(item)
        result = {"inputs": bundled_inputs, "responses": responses}
        if prompts: result["prompts"] = prompts
        return result
    # --- End: Original Framework Methods ---


    # --- Start: participant_model -> Your Entry Point ---
    # This function is called by the original _batched_inference
    def participant_model(self, prompt, qa_type=None): # Original signature + qa_type
        """
        This method is called by the evaluation framework's batching logic.
        It acts as the entry point to your custom RAG pipeline (_custom_pipeline_orchestrator).
        """
        # Infer qa_type if not provided (fragile)
        if qa_type is None:
            qa_type = "unknown"
            prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
            if isinstance(prompt, str): # Only infer if single prompt
                # Use specific phrases from templates for better inference
                if "multiple-choice question" in prompt_lower and "full text" in prompt_lower: qa_type = "multiple_choice"
                elif "list-based question" in prompt_lower and "comma-separated list of the full correct options" in prompt_lower : qa_type = "list"
                elif "true/false question" in prompt_lower: qa_type = "true_false"
                elif "incorrect explanation:" in prompt_lower and "short answer question" in prompt_lower: qa_type = "short_inverse"
                elif "short answer question" in prompt_lower: qa_type = "short"
                elif "incorrect reasoning step:" in prompt_lower and "multi-step question" in prompt_lower: qa_type = "multi_hop_inverse"
                elif "multi-step question" in prompt_lower: qa_type = "multi_hop"
                if qa_type == "unknown": print(f"WARN: Could not infer qa_type in participant_model: {prompt[:100]}...", flush=True)

        # Handle batched input from _batched_inference
        if isinstance(prompt, list):
             responses = []
             for p_idx, p in enumerate(prompt):
                 current_qa_type = qa_type # Use batch type if known
                 # Call orchestrator for each prompt
                 responses.append(self._custom_pipeline_orchestrator(p, current_qa_type, None))
             return responses
        else: # Handle single prompt input
             # Call orchestrator for the single prompt
             return self._custom_pipeline_orchestrator(prompt, qa_type, None)
    # --- End: participant_model ---


    # --- Start: Your Custom Helper Functions ---
    # (Paste all your helper methods _set_llm_max_lengths, _parse_question_from_prompt,
    #  _build_context_text, _retrieve_and_rerank, _apply_chat_template,
    #  _run_llm_generation, _verify_context, _generate_feedback_query,
    #  _generate_answer, _check_hallucination_and_refine,
    #  and _custom_pipeline_orchestrator here from the previous correct version)
    # Example of the orchestrator structure:
    def _custom_pipeline_orchestrator(self, prompt: str, qa_type: str, expected_answer_debug=None) -> str:
        if self.DEBUG:
             print(f"\n\n================ QA Item Start: Type = {qa_type} ================", flush=True)
             # print(f"DEBUG: Original Full Prompt (Orchestrator): {prompt}", flush=True) # Can be very long
             # if expected_answer_debug is not None: print(f"DEBUG: Expected: {expected_answer_debug}", flush=True)
        if not self.participant_pipelines: print("Orchestrator Error: Pipelines not loaded.", flush=True); return "Error: Pipelines unavailable."
        original_question = self._parse_question_from_prompt(prompt, qa_type)
        if self.DEBUG: print(f"DEBUG Orchestrator Parsed Q: '{original_question}'", flush=True)

        final_answer_raw = ""
        # --- CONDITIONAL LOGIC based on qa_type ---
        if qa_type in ["true_false", "multiple_choice", "list"]:
            if self.DEBUG: print(f"DEBUG Orchestrator: Direct Gen for {qa_type}.", flush=True)
            final_answer_raw = self._generate_answer(original_question, [], qa_type, prompt)
        elif qa_type in ["short_inverse", "multi_hop_inverse"]:
             if self.DEBUG: print(f"DEBUG Orchestrator: Inverse Task Gen for {qa_type}.", flush=True)
             try:
                 format_marker = "Expected response output format:"; idx = prompt.find(format_marker)
                 base_prompt_content = prompt[:idx].strip() if idx != -1 else prompt
                 if qa_type == "short_inverse":
                     inverse_instruction = "Analyze the 'Provided Answer' in the text above based on the 'Question'. Explain concisely why the 'Provided Answer' is incorrect, starting *only* with 'Incorrect Explanation:'. Do not add any other preamble or markdown."
                     max_tokens = 150; pipeline_key = 'generator_llm'; is_chat = True
                 else: # multi_hop_inverse
                     inverse_instruction = "Analyze the 'Full Reasoning' provided above for the 'Question' and identify the single step number containing the primary flaw leading to the incorrect 'Final Answer'. Output *only* the step number and a concise explanation using this exact format, with nothing before or after:\nIncorrect Reasoning Step: Step <number>\nIncorrect Reasoning Explanation: <explanation>"
                     max_tokens = 300; pipeline_key = 'generator_llm'; is_chat = True
                 inverse_content = f"{base_prompt_content}\n\n{inverse_instruction}"
             except Exception as e: print(f"Error building inverse prompt: {e}"); inverse_content = prompt; max_tokens=200; pipeline_key='generator_llm'; is_chat=True
             if self.DEBUG: print(f"\nDEBUG Orchestrator: Inverse Task Content ({pipeline_key}):\n---\n{inverse_content[:500]}...\n---\n", flush=True)
             final_answer_raw = self._run_llm_generation(pipeline_key, inverse_content, is_chat_model=is_chat, max_tokens=max_tokens)
             if self.DEBUG: print(f"DEBUG Orchestrator: Inverse Task Raw Output ({pipeline_key}): {final_answer_raw}", flush=True)
        elif qa_type in ["short", "multi_hop"]:
            if self.DEBUG: print(f"DEBUG Orchestrator: RAG Pipeline for {qa_type}.", flush=True)
            final_context_docs = []; current_query = original_question; max_loops = 3; initial_retrieval_failed = False
            for loop_count in range(max_loops):
                if self.DEBUG: print(f"\nDEBUG: RAG Loop Iteration: {loop_count + 1}/{max_loops}", flush=True)
                retrieved_docs = self._retrieve_and_rerank( current_query, top_k_initial=30, top_k_final=10 )
                if not retrieved_docs and loop_count == 0: print("Warning: Initial retrieval failed.", flush=True); initial_retrieval_failed = True; final_context_docs = []; break
                relevant_docs_from_step, sufficient_check_step = self._verify_context(current_query, retrieved_docs)
                existing_ids = {doc.get('id') for doc in final_context_docs if doc.get('id')}; newly_added_count = 0
                for doc in relevant_docs_from_step:
                    doc_id = doc.get('id'); is_duplicate = bool(doc_id and doc_id in existing_ids)
                    if not is_duplicate: final_context_docs.append(doc);
                    if doc_id: existing_ids.add(doc_id); newly_added_count += 1
                if self.DEBUG: print(f"DEBUG: Added {newly_added_count} new docs. Total relevant: {len(final_context_docs)}", flush=True)
                _, is_sufficient = self._verify_context(original_question, final_context_docs)
                if is_sufficient:
                    if self.DEBUG: print("DEBUG: Context sufficient. Breaking loop.", flush=True); break
                elif loop_count < max_loops - 1:
                    if self.DEBUG: print("DEBUG: Context insufficient, generating feedback query...", flush=True)
                    feedback_query = self._generate_feedback_query(original_question, final_context_docs)
                    if feedback_query == current_query or feedback_query == original_question:
                         if loop_count > 0 or current_query == original_question:
                              if self.DEBUG: print("DEBUG Warning: Feedback query failed/repeated. Breaking loop.", flush=True); break
                    current_query = feedback_query
                else: # Last loop
                    if self.DEBUG: print("DEBUG: Max feedback loops reached, proceeding.", flush=True)
            if initial_retrieval_failed: final_answer_raw = "Error: No relevant context found."
            else:
                 generated_answer = self._generate_answer(original_question, final_context_docs, qa_type, prompt)
                 if final_context_docs and "Error:" not in generated_answer: final_answer_raw = self._check_hallucination_and_refine(original_question, final_context_docs, generated_answer, qa_type, prompt)
                 else: final_answer_raw = generated_answer
        else: final_answer_raw = f"Error: Unknown QA type '{qa_type}' received by orchestrator."

        # Final Formatting (Keep enhanced version)
        final_answer_formatted = final_answer_raw
        try:
            if qa_type == "true_false":
                if re.search(r'\btrue\b', final_answer_raw, re.IGNORECASE): final_answer_formatted = "True"
                elif re.search(r'\bfalse\b', final_answer_raw, re.IGNORECASE): final_answer_formatted = "False"
                else: final_answer_formatted = "False"
            elif qa_type == "multiple_choice":
                match = re.search(r'\b([A-D])\b', final_answer_raw.upper());
                if match: final_answer_formatted = match.group(1)
                else: final_answer_formatted = ""; # Return empty if no letter found
            elif qa_type == "list":
                 letters = re.findall(r'\b([A-Z])\b', final_answer_raw.upper()); final_answer_formatted = ", ".join(sorted(list(set(letters))));
                 if not letters: final_answer_formatted = ""
            elif qa_type == "short":
                 match = re.search(r"Final Answer:(.*)", final_answer_raw, re.IGNORECASE | re.DOTALL); ans_part = match.group(1).strip() if match else final_answer_raw
                 final_answer_formatted = f"Final Answer: {ans_part}"
                 words = ans_part.split();
                 if len(words) > 110: final_answer_formatted = "Final Answer: " + " ".join(words[:100]) + "..."
            elif qa_type == "multi_hop":
                ans_marker = "Final Answer:"; res_marker = "Reasoning:"
                ans_part = final_answer_raw; res_part = "[No reasoning provided]"
                res_match = re.search(f"{res_marker}(.*)", final_answer_raw, re.IGNORECASE | re.DOTALL)
                if res_match: res_part = res_match.group(1).strip(); ans_part = final_answer_raw[:res_match.start()].strip()
                ans_match = re.search(f"{ans_marker}(.*)", ans_part, re.IGNORECASE | re.DOTALL)
                if ans_match: ans_part = ans_match.group(1).strip()
                final_answer_formatted = f"{ans_marker} {ans_part}\n{res_marker} {res_part}"
            elif qa_type == "short_inverse":
                exp_marker = "Incorrect Explanation:";
                match = re.search(f"{exp_marker}(.*)", final_answer_raw, re.IGNORECASE | re.DOTALL)
                if match: exp_part = match.group(1).strip()
                else: exp_part = final_answer_raw
                final_answer_formatted = f"{exp_marker} {exp_part}"
            elif qa_type == "multi_hop_inverse":
                 step_marker = "Incorrect Reasoning Step:"; exp_marker = "Incorrect Reasoning Explanation:"
                 step_part = "Step ?"; exp_part = "[No explanation provided]"
                 step_match = re.search(r"(?:Incorrect Reasoning Step:?|Step)\s*(\d+)", final_answer_raw, re.IGNORECASE)
                 if step_match: step_part = f"Step {step_match.group(1)}"
                 exp_match = re.search(f"{exp_marker}(.*)", final_answer_raw, re.IGNORECASE | re.DOTALL)
                 if exp_match: exp_part = exp_match.group(1).strip()
                 elif step_part != "Step ?" and step_match: exp_part = final_answer_raw[step_match.end():].strip().replace(exp_marker, "").strip()
                 elif not step_match : exp_part = final_answer_raw.replace(step_marker,"").strip()
                 final_answer_formatted = f"{step_marker} {step_part}\n{exp_marker} {exp_part}"
        except Exception as fmt_err: print(f"ERROR during final format enforcement: {fmt_err}.", flush=True); final_answer_formatted = final_answer_raw # Fallback
        if self.DEBUG: print(f"DEBUG Orchestrator: Final Formatted Answer: {final_answer_formatted}", flush=True)
        if self.DEBUG: print(f"================ QA Item End: Type = {qa_type} ===============", flush=True)
        return final_answer_formatted
        # --- End: Orchestrator Logic ---

    # --- Start: Original Submission Runners (Unchanged) ---
    def submit_true_false_questions(self): # Original
        try:
            tf_data = self.sampled_qa_pairs.get("true_false", [])
            if not tf_data: print("No TF data loaded.", flush=True); return {"responses": [], "inputs": []}
            template = self.load_template("tf_template.prompt"); prompts = []
            for qa in tf_data:
                try: prompts.append(self.generate_prompt(template, qa, "true_false"))
                except Exception as e: print(f"Error generating prompt TF: {e}", flush=True); prompts.append("")
            try: responses = self._batched_inference(prompts, qa_type="true_false")
            except Exception as e: print(f"Error during inference TF: {e}", flush=True); responses = ["ERROR"] * len(prompts)
            return self._bundle(tf_data, responses, prompts)
        except Exception as e: print(f"Error submitting TF: {e}", flush=True); return {"responses": [], "inputs": [], "prompts": []}

    def submit_multiple_choice_questions(self): # Original
        try:
            mc_data = self.sampled_qa_pairs.get("multiple_choice", [])
            if not mc_data: print("No MC data loaded.", flush=True); return {"responses": [], "inputs": [], "prompts": []}
            template = self.load_template("MC_template.prompt"); prompts = []
            for qa in mc_data:
                try: prompts.append(self.generate_prompt(template, qa, "multiple_choice"))
                except Exception as e: print(f"Error generating prompt MC: {e}", flush=True); prompts.append("")
            try: responses = self._batched_inference(prompts, qa_type="multiple_choice")
            except Exception as e: print(f"Error during inference MC: {e}", flush=True); responses = ["ERROR"] * len(prompts)
            return self._bundle(mc_data, responses, prompts)
        except Exception as e: print(f"Error submitting MC: {e}", flush=True); return {"responses": [], "inputs": [], "prompts": []}

    def submit_list_questions(self): # Original
        try:
            list_data = self.sampled_qa_pairs.get("list", [])
            if not list_data: print("No List data loaded.", flush=True); return {"responses": [], "inputs": [], "prompts": []}
            template = self.load_template("list_template.prompt"); prompts = []
            for qa in list_data:
                try: prompts.append(self.generate_prompt(template, qa, "list"))
                except Exception as e: print(f"Error generating prompt List: {e}", flush=True); prompts.append("")
            try: responses = self._batched_inference(prompts, qa_type="list")
            except Exception as e: print(f"Error during inference List: {e}", flush=True); responses = ["ERROR"] * len(prompts)
            return self._bundle(list_data, responses, prompts)
        except Exception as e: print(f"Error submitting List: {e}", flush=True); return {"responses": [], "inputs": [], "prompts": []}

    def submit_short_questions(self): # Original
        try:
            short_data = self.sampled_qa_pairs.get("short", [])
            if not short_data: print("No Short data loaded.", flush=True); return {"responses": [], "inputs": [], "prompts": []}
            template = self.load_template("short_template.prompt"); prompts = []
            for qa in short_data:
                try: prompts.append(self.generate_prompt(template, qa, "short"))
                except Exception as e: print(f"Error generating prompt Short: {e}", flush=True); prompts.append("")
            try: responses = self._batched_inference(prompts, qa_type="short")
            except Exception as e: print(f"Error during inference Short: {e}", flush=True); responses = ["ERROR"] * len(prompts)
            return self._bundle(short_data, responses, prompts)
        except Exception as e: print(f"Error submitting Short: {e}", flush=True); return {"responses": [], "inputs": [], "prompts": []}

    def submit_short_inverse_questions(self): # Original
        try:
            short_inverse_data = self.sampled_qa_pairs.get("short_inverse", [])
            if not short_inverse_data: print("No Short Inverse data loaded.", flush=True); return {"responses": [], "inputs": [], "prompts": []}
            template = self.load_template("short_inverse_template.prompt"); prompts = []
            for qa in short_inverse_data:
                try: prompts.append(self.generate_prompt(template, qa, "short_inverse"))
                except Exception as e: print(f"Error generating prompt ShortInv: {e}", flush=True); prompts.append("")
            try: responses = self._batched_inference(prompts, qa_type="short_inverse")
            except Exception as e: print(f"Error during inference ShortInv: {e}", flush=True); responses = ["ERROR"] * len(prompts)
            return self._bundle(short_inverse_data, responses, prompts)
        except Exception as e: print(f"Error submitting Short Inverse: {e}", flush=True); return {"responses": [], "inputs": [], "prompts": []}

    def submit_multi_hop_questions(self): # Original
        try:
            mh_data = self.sampled_qa_pairs.get("multi_hop", [])
            if not mh_data: print("No MH data loaded.", flush=True); return {"responses": [], "inputs": [], "prompts": []}
            template = self.load_template("multi_hop_template.prompt"); prompts = []
            for qa in mh_data:
                try: prompts.append(self.generate_prompt(template, qa, "multi_hop"))
                except Exception as e: print(f"Error generating prompt MH: {e}", flush=True); prompts.append("")
            try: responses = self._batched_inference(prompts, qa_type="multi_hop")
            except Exception as e: print(f"Error during inference MH: {e}", flush=True); responses = ["ERROR"] * len(prompts)
            return self._bundle(mh_data, responses, prompts)
        except Exception as e: print(f"Error submitting MH: {e}", flush=True); return {"responses": [], "inputs": [], "prompts": []}

    def submit_multi_hop_inverse_questions(self): # Original
        try:
            mh_inverse_data = self.sampled_qa_pairs.get("multi_hop_inverse", [])
            if not mh_inverse_data: print("No MH Inverse data loaded.", flush=True); return {"responses": [], "inputs": [], "prompts": []}
            template = self.load_template("multi_hop_inverse_template.prompt"); prompts = []
            for qa in mh_inverse_data:
                try: prompts.append(self.generate_prompt(template, qa, "multi_hop_inverse"))
                except Exception as e: print(f"Error generating prompt MHInv: {e}", flush=True); prompts.append("")
            try: responses = self._batched_inference(prompts, qa_type="multi_hop_inverse")
            except Exception as e: print(f"Error during inference MHInv: {e}", flush=True); responses = ["ERROR"] * len(prompts)
            return self._bundle(mh_inverse_data, responses, prompts)
        except Exception as e: print(f"Error submitting MH Inverse: {e}", flush=True); return {"responses": [], "inputs": [], "prompts": []}

    def run_all_submissions(self): # Original
        try:
            # Original saves output to /app/submission_output
            # This path should be accessible inside the container
            output_dir = os.path.join(self.base_dir, "submission_output")
            os.makedirs(output_dir, exist_ok=True)
            qa_types = { "true_false": self.submit_true_false_questions, "multiple_choice": self.submit_multiple_choice_questions, "list": self.submit_list_questions, "short": self.submit_short_questions, "multi_hop": self.submit_multi_hop_questions, "short_inverse": self.submit_short_inverse_questions, "multi_hop_inverse": self.submit_multi_hop_inverse_questions, }
            for qa_type, submit_fn in qa_types.items():
                print(f"Running inference for: {qa_type}", flush=True)
                result = submit_fn()
                output_path = os.path.join(output_dir, f"{qa_type}.json")
                with open(output_path, "w") as f: json.dump(result, f, indent=4)
                print(f"Saved {qa_type} results to {output_path}", flush=True)
            print(f"All inference outputs saved to separate JSON files in {output_dir}", flush=True)
        except Exception as e: print(f"Error running all submissions: {e}", flush=True)
    # --- End: Original Methods ---

# --- Start: Original __main__ block ---
def parse_args():
    parser = argparse.ArgumentParser(description="ClinIQLink submission Script")
    parser.add_argument( "--mode", choices=["local", "container"], default="container", help="Run mode")
    parser.add_argument( "--max_length", type=int, default=1028, help="Max output token length (Argument likely unused if generation params are hardcoded/dynamic)")
    parser.add_argument("--num_tf", type=int, default=200, help="Num True/False Qs")
    parser.add_argument("--num_mc", type=int, default=200, help="Num Multiple Choice Qs")
    parser.add_argument("--num_list", type=int, default=200, help="Num List Qs")
    parser.add_argument("--num_short", type=int, default=200, help="Num Short Answer Qs")
    parser.add_argument("--num_short_inv", type=int, default=200, help="Num Short Inverse Qs")
    parser.add_argument("--num_multi", type=int, default=200, help="Num Multi-hop Qs")
    parser.add_argument("--num_multi_inv", type=int, default=200, help="Num Multi-hop Inverse Qs")
    parser.add_argument("--random", action="store_true", help="Randomly sample Qs")
    parser.add_argument("--chunk_size", type=int, default=2, help="Inference batch size")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sample_sizes = { "num_tf": args.num_tf, "num_mc": args.num_mc, "num_list": args.num_list, "num_short": args.num_short, "num_short_inv": args.num_short_inv, "num_multi": args.num_multi, "num_multi_inv": args.num_multi_inv, }
    submit = ClinIQLinkSampleDatasetSubmit(
        run_mode=args.mode, max_length=args.max_length, sample_sizes=sample_sizes,
        random_sample=args.random, chunk_size=args.chunk_size,
        do_sample=args.do_sample, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k,
        debug_mode=True # Force debug ON for testing
    )
    submit.run_all_submissions() # Original entry point
    print("ClinIQLink Submission Script Finished.", flush=True)
# --- End: submit.py ---