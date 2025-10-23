

import sys
import logging
import argparse
from cyoa.agent_orchestrator import AgentOrchestrator, get_user_character_info


def main():

    parser = argparse.ArgumentParser(description="LLM CYOA")
    parser.add_argument('--debug', action='store_true', help='Print debug info to console as well as log')
    args = parser.parse_args()

    handlers = [logging.FileHandler("cyoa_debug.log")]
    if args.debug:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger("cyoa")
    logger.setLevel(logging.DEBUG)

    print("Welcome to LLM CYOA!")
    print("Type your actions or dialogue. Type 'quit' to exit.")
    print("Example: 'Astra Vey asks Kael about the city.' or 'Astra Vey draws her sword.'\n")

    # Get user character info interactively
    user_name, user_background = get_user_character_info()
    print(f"\nYour character: {user_name}\nBackground: {user_background}\n")

    # Model and GPU config (adjust as needed)
    model_path = "meta-llama/Llama-3.2-3B-Instruct"
    orchestrator = AgentOrchestrator(
        model_path,
        storyteller_gpu=0,
        director_gpu=1,
        character_gpu=2
    )
    orchestrator.set_logger(logger)
    orchestrator.start_storyteller_and_director()

    user_inputs = []
    turn = 0
    max_turns = 5
    try:
        # Generate and display the story introduction before prompting the user
        print("\n[Progress] Generating story introduction...")
        if args.debug:
            orig_print = __builtins__.print
            def debug_print(*args, **kwargs):
                logger.debug(' '.join(str(a) for a in args))
            __builtins__.print = debug_print
        try:
            # Build initial storyteller prompt for intro using orchestrator method
            storyteller_prompt = orchestrator.build_storyteller_prompt_with_user(user_name, user_background)
            director_prompt = orchestrator.build_director_prompt("", user_name)
            story, _, _ = orchestrator.run_story_agents(storyteller_prompt, director_prompt)
        finally:
            if args.debug:
                __builtins__.print = orig_print
        print(f"\n[Story Update]\n{story}\n")
        # Now enter the user input loop
        while turn < max_turns:
            user_input = input(f"\n--- Turn {turn+1} ---\nWhat does {user_name} do or say? ").strip()
            if user_input.lower() == 'quit':
                print("Exiting story.")
                break
            user_inputs.append(user_input)
            print("\n[Progress] Generating story...")
            if args.debug:
                orig_print = __builtins__.print
                def debug_print(*args, **kwargs):
                    logger.debug(' '.join(str(a) for a in args))
                __builtins__.print = debug_print
            try:
                # Build prompts for each user turn
                user_input_latest = user_inputs[-1]
                storyteller_prompt = [
                    {"role": "system", "content": orchestrator.STORYTELLER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input_latest}
                ]
                director_prompt = orchestrator.build_director_prompt(user_input_latest, user_name)
                story, _, _ = orchestrator.run_story_agents(storyteller_prompt, director_prompt)
            finally:
                if args.debug:
                    __builtins__.print = orig_print
            print(f"\n[Story Update]\n{story}\n")
            turn += 1
    except KeyboardInterrupt:
        print("\nSession interrupted. Exiting gracefully...")
    finally:
        orchestrator.stop_all()
        print("\nThanks for playing!")

if __name__ == "__main__":
    main()
