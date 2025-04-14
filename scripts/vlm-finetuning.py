#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen 2.5VL Fine-tuning Script
-----------------------------
An efficient implementation for fine-tuning Qwen 2.5VL on autonomous driving data.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Union, Any, Optional
from datetime import datetime

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
import torch
import wandb
from tqdm import tqdm
from datasets import Dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_CONFIG = {
    "model_id": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
    "data_path": "drivelm_with_thinking_800_frames_clean.json",
    "base_image_path": "../challenge/data/nuscenes",
    "output_dir": "outputs_Qwen2.5-VL-7B-Instruct",
    "epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 12,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 50,
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0,
    "seed": 3407,
    "max_seq_length": 2048,
    "test_size": 0.1,
    "logging_steps": 10,
    "eval_steps": 10,
    "save_steps": 50,
    "save_total_limit": 3,
    "use_wandb": True,
    "wandb_project": "qwen-vl-driving",
    "wandb_run_name": None
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5VL for autonomous driving tasks")
    
    parser.add_argument("--model_id", type=str, default=DEFAULT_CONFIG["model_id"],
                        help="Model ID to use for fine-tuning")
    parser.add_argument("--data_path", type=str, default=DEFAULT_CONFIG["data_path"],
                        help="Path to the dataset JSON file")
    parser.add_argument("--base_image_path", type=str, default=DEFAULT_CONFIG["base_image_path"],
                        help="Base path to the image directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"],
                        help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"],
                        help="Number of warmup steps")
    parser.add_argument("--lora_r", type=int, default=DEFAULT_CONFIG["lora_r"],
                        help="LoRA rank (r) value")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"],
                        help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_CONFIG["lora_dropout"],
                        help="LoRA dropout rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"],
                        help="Random seed for reproducibility")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG["max_seq_length"],
                        help="Maximum sequence length")
    parser.add_argument("--test_size", type=float, default=DEFAULT_CONFIG["test_size"],
                        help="Proportion of the dataset to use for testing")
    parser.add_argument("--use_wandb", action="store_true", default=DEFAULT_CONFIG["use_wandb"],
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_CONFIG["wandb_project"],
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=DEFAULT_CONFIG["wandb_run_name"],
                        help="Weights & Biases run name (defaults to timestamp if not provided)")
    
    return parser.parse_args()

class QwenVLTrainer:
    """Trainer class for fine-tuning Qwen 2.5VL models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with the given configuration.
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.trainer = None
        
        # Set seed for reproducibility
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        
        # Timestamp for output directory and run naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config["model_id"].split('/')[-1]
        data_name = Path(config["data_path"]).stem
        self.output_dir = Path(config["output_dir"]) / f"{model_name}_{data_name}_{timestamp}"
        
        # Initialize wandb if enabled
        if config["use_wandb"]:
            wandb_run_name = config["wandb_run_name"] or f"{model_name}_{data_name}_{timestamp}"
            logger.info(f"Initializing Weights & Biases with project: {config['wandb_project']}, run: {wandb_run_name}")
            wandb.init(
                project=config["wandb_project"],
                name=wandb_run_name,
                config=config
            )
        
        # Load model and tokenizer
        self._load_model()
        
        # Log GPU information
        if torch.cuda.is_available():
            self._log_gpu_info()
    
    def _load_model(self):
        """Load and prepare the model and tokenizer."""
        logger.info(f"Loading model: {self.config['model_id']}")
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.config["model_id"],
            load_in_4bit=True,  # Use 4bit quantization to reduce memory usage
            use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing for longer context
        )
        
        logger.info("Applying LoRA configuration")
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=False,  # Don't finetune vision layers to save memory
            finetune_language_layers=True,  # Finetune language layers
            finetune_attention_modules=False,  # Don't finetune attention layers to save memory
            finetune_mlp_modules=True,  # Finetune MLP layers
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            random_state=self.config["seed"],
            use_rslora=False,
            loftq_config=None,
        )
    
    def _log_gpu_info(self):
        """Log GPU information for monitoring."""
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logger.info(f"{start_gpu_memory} GB of memory reserved.")
        else:
            logger.warning("No GPU detected! Training may be very slow.")
    
    def prepare_data(self):
        """Load and prepare the dataset for training."""
        logger.info(f"Loading dataset from: {self.config['data_path']}")
        
        # Check if data file exists
        if not os.path.exists(self.config["data_path"]):
            raise FileNotFoundError(f"Dataset file not found: {self.config['data_path']}")
        
        # Load dataset
        dataset = Dataset.from_json(self.config["data_path"])
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Split dataset
        splits = dataset.train_test_split(test_size=self.config["test_size"], seed=self.config["seed"])
        raw_train_dataset, raw_test_dataset = splits["train"], splits["test"]
        
        logger.info(f"Processing {len(raw_train_dataset)} training samples...")
        self.train_dataset = [self._format_sample(example) for example in tqdm(raw_train_dataset)]
        
        logger.info(f"Processing {len(raw_test_dataset)} test samples...")
        self.test_dataset = [self._format_sample(example) for example in tqdm(raw_test_dataset)]
        
        logger.info(f"Prepared {len(self.train_dataset)} training samples and "
                   f"{len(self.test_dataset)} test samples")
    
    def _fix_image_paths(self, image_paths: Union[str, List[str]]) -> List[str]:
        """
        Fix image paths by converting relative paths to absolute paths.
        
        Args:
            image_paths: Single image path or list of image paths
            
        Returns:
            List of fixed image paths
        """
        # Ensure we have a list
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        # Fix each path
        fixed_paths = []
        for path in image_paths:
            fixed_path = path.replace("../nuscenes", self.config["base_image_path"])
            
            # Convert to absolute path if it's not already
            if not os.path.isabs(fixed_path):
                fixed_path = os.path.abspath(fixed_path)
            
            # Check if the file exists
            if os.path.exists(fixed_path):
                fixed_paths.append(fixed_path)
            else:
                logger.warning(f"Image not found: {fixed_path}")
        
        return fixed_paths
    
    def _format_sample(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a single example for the model.
        
        Args:
            example: Raw example from the dataset
            
        Returns:
            Formatted example in the proper structure for Qwen2.5-VL
        """
        try:
            # Get image paths
            image_paths = self._fix_image_paths(example["image"])
            
            # Skip if no valid images
            if not image_paths:
                logger.warning(f"No valid images found for example, skipping")
                return None
            
            # Create the prompt
            prompt = (
                f"You are an AI assistant for autonomous driving. Analyze the scene and reason through driving decisions carefully.\n\n"
                f"Analyze the following driving scenario and provide reasoning:\n\n"
                f"{example['problem']}\n\n"
                f"Put your reasoning within <think></think> tags and your final answer within <answer></answer> tags."
            )
            
            # Get the thinking and solution
            thinking = example.get("thinking", "").strip()
            solution = example.get("solution", "").strip()
            
            # Combine thinking and solution for the completion
            completion = f"{thinking}\n\n{solution}"
            
            # Create content list in the format expected by Qwen2.5-VL
            content_list = []
            
            # Add all images first
            for img_path in image_paths:
                content_list.append({
                    "type": "image", 
                    "image": "file://" + img_path, 
                    'resized_height': 224,  # 5*28 (smaller height)
                    'resized_width': 224    # 9*28 (smaller width, maintains ~16:9 ratio)
                })
            
            # Add text prompt
            content_list.append({"type": "text", "text": prompt})
            
            # Format in the messages format for Qwen2.5-VL
            processed_example = {
                'messages': [
                    {"role": "user", "content": content_list},
                    {"role": "assistant", "content": [{"type": "text", "text": completion}]},
                ]
            }
            
            return processed_example
            
        except Exception as e:
            logger.error(f"Error processing example: {str(e)}")
            return None
    
    # Add early stopping callback
    def setup_trainer(self):
        """Set up the SFT Trainer for fine-tuning."""
        logger.info("Setting up trainer")
        
        # Prepare model for training
        FastVisionModel.for_training(self.model)
        
        # Build output directory path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up trainer with early stopping
        from transformers import EarlyStoppingCallback
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer, max_seq_length=4096),  # Required for vision models
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            args=SFTConfig(
                per_device_train_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                warmup_steps=self.config["warmup_steps"],
                num_train_epochs=self.config["epochs"],
                learning_rate=self.config["learning_rate"],
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=self.config["logging_steps"],
                optim="paged_adamw_8bit",
                weight_decay=self.config["weight_decay"],
                lr_scheduler_type="linear",
                seed=self.config["seed"],
                output_dir=str(self.output_dir),
                report_to="wandb" if self.config["use_wandb"] else "none",
                
                # Memory optimization options
                gradient_checkpointing=True,
                ddp_find_unused_parameters=False,
                dataloader_num_workers=0,  # Reduces CPU memory usage
                dataloader_pin_memory=False,  # Reduces CPU->GPU transfer overhead
                torch_compile=False,  # Can improve speed but uses more memory during compilation
                
                # Required settings for vision finetuning
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=4,
                max_seq_length=self.config["max_seq_length"],
                
                # Evaluation and saving strategies must match for load_best_model_at_end
                evaluation_strategy="steps",  # Match with save_strategy
                eval_steps=self.config["eval_steps"],
                save_strategy="steps", 
                save_steps=self.config["save_steps"],
                save_total_limit=self.config["save_total_limit"],
                
                # Early stopping - save best model
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
    
    def train(self):
        """Run the training process."""
        logger.info("Starting training")
        
        try:
            # Run training
            trainer_stats = self.trainer.train()
            
            # Log training stats
            logger.info(f"Training completed. Stats: {trainer_stats}")
            
            # Save the model
            logger.info(f"Saving model to {self.output_dir}")
            self.trainer.save_model(str(self.output_dir))
            
            # Log model metadata to wandb if enabled
            if self.config["use_wandb"]:
                # Log final metrics
                wandb.log({"final_loss": trainer_stats.training_loss})
                
                # Save model as artifact
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}", 
                    type="model",
                    description=f"Fine-tuned {self.config['model_id']} model"
                )
                artifact.add_dir(str(self.output_dir))
                wandb.log_artifact(artifact)
                
                # Finish the wandb run
                wandb.finish()
            
            return trainer_stats
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            if self.config["use_wandb"]:
                wandb.finish(exit_code=1)
            raise
    
    def run_inference_example(self):
        """Run inference on a single example from the test dataset."""
        logger.info("Running inference on a test example")
        
        # Prepare model for inference
        FastVisionModel.for_inference(self.model)
        
        try:
            # Get the first example from test dataset
            example = self.test_dataset[0]
            if 'Qwen' in self.config['model_id']:
                # Import the vision processing utility
                from qwen_vl_utils import process_vision_info
                
                # Process vision information
                image_inputs, video_inputs = process_vision_info([example["messages"][0]])
            else:
                from PIL import Image as PILImage
                image_inputs = []
                for path in [im_path['image'] for im_path in example["messages"][0]['content'][:6]]:
                    img = PILImage.open(path[6:]).resize((224, 224))
                    image_inputs.append(img)
            
            # Prepare input text
            input_text = self.tokenizer.apply_chat_template([example["messages"][0]], add_generation_prompt=True)
            
            # Tokenize inputs
            inputs = self.tokenizer(
                image_inputs,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Set up text streamer
            text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            # Generate text
            outputs = self.model.generate(
                **inputs, 
                streamer=text_streamer,
                max_new_tokens=128,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )
            
            # Get and return the generated text
            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            logger.info(f"Generated response: {generated_text}")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return None

def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Add to config dictionary
    config = {
        "model_id": args.model_id,
        "data_path": args.data_path,
        "base_image_path": args.base_image_path,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "test_size": args.test_size,
        "logging_steps": DEFAULT_CONFIG["logging_steps"],
        "eval_steps": DEFAULT_CONFIG["eval_steps"],
        "save_steps": DEFAULT_CONFIG["save_steps"],
        "save_total_limit": DEFAULT_CONFIG["save_total_limit"],
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name
    }
    
    logger.info("Starting Qwen 2.5VL fine-tuning")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Initialize trainer
        trainer = QwenVLTrainer(config)
        
        # Prepare dataset
        trainer.prepare_data()
        
        # Setup trainer
        trainer.setup_trainer()
        
        # Run training
        trainer.train()
        
        # Run inference example
        trainer.run_inference_example()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()