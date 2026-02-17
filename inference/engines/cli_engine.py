"""
Command-line interface for LEMA chatbot.
"""

import sys
import os
from pathlib import Path

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.framework.model_handler import LemaModelHandler
from inference.framework.chat_parser import ChatParser
from inference.framework.conversation import ConversationManager

class CLIChatEngine:
    """Interactive CLI chat interface."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize CLI chat engine.
        
        Args:
            checkpoint_path: Path to LEMA checkpoint
            device: Device to run on
        """
        print("Loading model...")
        self.model_handler = LemaModelHandler(checkpoint_path, device)
        self.conversation = ConversationManager()
        self.parser = ChatParser()
        print("✅ Model loaded!\n")
    
    def run(self):
        """Run interactive chat loop."""
        print("=" * 60)
        print("LEMA Chatbot - Custom Format Demonstration")
        print("=" * 60)
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit the chat")
        print("  'clear' - Clear conversation history")
        print("  'debug' - Toggle debug mode")
        print("\n" + "=" * 60 + "\n")
        
        debug_mode = False
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation.clear()
                    print("\n🔄 Conversation cleared\n")
                    continue
                
                if user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"\n🐛 Debug mode: {'ON' if debug_mode else 'OFF'}\n")
                    continue
                
                # Add to conversation
                self.conversation.add_user_message(user_input)
                
                # Format prompt
                prompt = self.parser.format_prompt(user_input)
                
                if debug_mode:
                    print(f"\n[DEBUG] Prompt:\n{prompt}\n")
                
                # Generate response
                response_text = self.model_handler.generate(prompt)
                
                if debug_mode:
                    print(f"\n[DEBUG] Raw response:\n{response_text}\n")
                
                # Parse response
                parsed = self.parser.parse_response(response_text)
                
                if parsed.is_valid:
                    # Display parsed response
                    print(f"\nAssistant: {parsed.answer}")
                    print(f"💡 {parsed.explanation}")
                    print(f"📊 Confidence: {parsed.confidence}\n")
                    
                    # Add to conversation
                    self.conversation.add_assistant_message(
                        parsed.answer,
                        explanation=parsed.explanation,
                        confidence=parsed.confidence
                    )
                else:
                    # Model didn't follow format
                    print(f"\n⚠️  Model response didn't follow LEMA format:")
                    print(f"{response_text}\n")
                    print("This might indicate the model needs more training.\n")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                if debug_mode:
                    import traceback
                    traceback.print_exc()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LEMA CLI Chat Interface")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to LEMA checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    engine = CLIChatEngine(args.checkpoint, args.device)
    engine.run()

if __name__ == "__main__":
    main()
