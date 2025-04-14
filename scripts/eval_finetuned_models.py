import os
import json
import argparse

from tqdm import tqdm
import torch
from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info
from transformers import TextStreamer
from datasets import Dataset

def evaluate_model(model_path, eval_dataset, base_image_path="../challenge/data/nuscenes", output_file="model_responses.json"):
    """
    Load a fine-tuned model using unsloth and generate responses for the evaluation dataset.
    Uses the exact same data formatting approach as in training.
    
    Args:
        model_path: Path to the fine-tuned model
        eval_dataset: Dataset object containing evaluation examples
        base_image_path: Base path to the image directory
        output_file: Path to save the evaluation results
    """
    print(f"Loading model from: {model_path}")
    
    # Load the model using unsloth
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        # use_cache=True,
    )
    
    # Prepare model for inference
    FastVisionModel.for_inference(model)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    
    # List to store all evaluation results
    results = []
    
    # Function to fix image paths
    def fix_image_paths(image_paths):
        # Ensure we have a list
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        # Fix each path
        fixed_paths = []
        for path in image_paths:
            fixed_path = path.replace("../nuscenes", base_image_path)
            
            # Convert to absolute path if it's not already
            if not os.path.isabs(fixed_path):
                fixed_path = os.path.abspath(fixed_path)
            
            # Check if the file exists
            if os.path.exists(fixed_path):
                fixed_paths.append(fixed_path)
            else:
                print(f"Warning: Image not found: {fixed_path}")
        
        return fixed_paths
    
    # Function to format a sample exactly like in training
    def format_sample(example):
        """
        Format a single example for the model.
        """
        try:
            # Get image paths
            image_paths = fix_image_paths(example["image"])
            
            # Skip if no valid images
            if not image_paths:
                print(f"Warning: No valid images found for example, skipping")
                return None
            
            # Create the prompt
            prompt = (
                f"You are an AI assistant for autonomous driving. Analyze the scene and reason through driving decisions carefully.\n\n"
                f"Analyze the following driving scenario and provide reasoning:\n\n"
                f"{example['problem']}\n\n"
                f"Put your reasoning within <think></think> tags and your final answer within <answer></answer> tags."
            )
            
            # Create content list in the format expected by Qwen2.5-VL
            content_list = []
            
            # Add all images first
            for img_path in image_paths:
                content_list.append({
                    "type": "image", 
                    "image": "file://" + img_path, 
                    'resized_height': 224,
                    'resized_width': 224
                })
            
            # Add text prompt
            content_list.append({"type": "text", "text": prompt})
            
            # Format in the messages format for Qwen2.5-VL
            processed_example = {
                'messages': [
                    {"role": "user", "content": content_list},
                    # No assistant message for inference
                ]
            }
            
            return processed_example
            
        except Exception as e:
            print(f"Error formatting example: {str(e)}")
            return None
    
    # Process each example in the evaluation dataset
    for i, example in enumerate(tqdm(eval_dataset, desc="Evaluating examples")):
        try:
            # Extract metadata and create unique ID
            metadata = example.get("metadata", {})
            scene_id = metadata.get("scene_id", "unknown")
            frame_id = metadata.get("frame_id", "unknown")
            idx = i  # Use iteration index if no other index is available            
            unique_id = f"{scene_id}_{frame_id}_{idx}"
            
            # Get question and ground truth answer
            question = example.get("problem", "")
            
            # Extract ground truth answer from solution field
            solution = example.get("solution", "")
            if "<answer>" in solution and "</answer>" in solution:
                start_idx = solution.find("<answer>") + len("<answer>")
                end_idx = solution.find("</answer>")
                gt_answer = solution[start_idx:end_idx].strip()
            else:
                gt_answer = solution
            
            # Format the example exactly like in training
            formatted_example = format_sample(example)
            
            if formatted_example is None:
                print(f"Skipping example {unique_id} due to formatting error")
                continue
            
            # Process vision information
            image_inputs, video_inputs = process_vision_info([formatted_example["messages"][0]])
            
            # Prepare input text
            input_text = tokenizer.apply_chat_template([formatted_example["messages"][0]], add_generation_prompt=True)
            
            # Create inputs
            inputs = tokenizer(
                image_inputs,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            # Decode the outputs
            response = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # Create result entry
            result = {
                "id": unique_id,
                "question": f"<image>\n{question}",
                "gt_answer": gt_answer,
                "answer": response.strip()
            }
            
            # Add to results
            results.append(result)
            
            # Print progress every 10 examples
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(eval_dataset)} examples")
                
                # Save intermediate results
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved intermediate results to {output_file}")
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Processed {len(results)}/{len(eval_dataset)} examples.")
    print(f"Results saved to {output_file}")
    
    return results



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned VLM model on driving data")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--data_path", type=str, default="drivelm_with_thinking_800_frames_clean.json",
                        help="Path to the evaluation data JSON file")
    parser.add_argument("--base_image_path", type=str, default="../challenge/data/nuscenes",
                        help="Base path to the image directory")
    parser.add_argument("--output_path", type=str, default="model_responses.json",
                        help="Path to save evaluation results")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Test split size")
    parser.add_argument("--seed", type=int, default=3047,
                        help="Random seed for the data split")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.data_path}")
    
    # Load the dataset
    dataset = Dataset.from_json(args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Split dataset
    if args.test_size == 1:
        eval_dataset = dataset
    else:
        splits = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
        _, eval_dataset = splits["train"], splits["test"]
    print(f"Evaluation dataset contains {len(eval_dataset)} samples")
    
    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        eval_dataset=eval_dataset,
        base_image_path=args.base_image_path,
        output_file=args.output_path
    )
    
    print(f"Evaluation completed. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()