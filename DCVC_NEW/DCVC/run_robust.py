import os
import json
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Original full config")
    parser.add_argument('--model_i', type=str, required=True)
    parser.add_argument('--model_p', type=str, required=True)
    parser.add_argument('--base_model_p', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Load original config
    with open(args.config, 'r') as f:
        full_config = json.load(f)

    base_output_dir = args.output_dir
    bin_dir = os.path.join(base_output_dir, "out_bin")
    
    # Iterate over every sequence
    for ds_name, ds_content in full_config['test_classes'].items():
        if ds_content['test'] == 0: continue
        
        for seq_name, seq_info in ds_content['sequences'].items():
            print(f"\n[RobustRunner] >>> Preparing to run: {seq_name}")
            
            # 1. Create a temporary config for JUST this sequence
            single_seq_config = {
                "root_path": full_config.get('root_path', ''),
                "test_classes": {
                    ds_name: {
                        "base_path": ds_content['base_path'],
                        "src_type": ds_content['src_type'],
                        "test": 1,
                        "sequences": {
                            seq_name: seq_info
                        }
                    }
                }
            }
            
            temp_config_name = f"temp_config_{seq_name}.json"
            with open(temp_config_name, 'w') as tf:
                json.dump(single_seq_config, tf, indent=2)

            # 2. Construct the output path specific to this sequence file
            # We use a specific filename so test_video.py writes exactly where we want
            # However, test_video.py aggregates results. We will let it write to a specific json
            # and later you can merge them, or just use compare_dcvc on the folder.
            seq_output_json = os.path.join(base_output_dir, f"{seq_name}_result.json")

            # 3. Build the command
            cmd = [
                "python", "test_video.py",
                "--test_config", temp_config_name,
                "--model_path_i", args.model_i,
                "--base_model_p", args.base_model_p,
                "--model_path_p", args.model_p,
                "--output_path", seq_output_json,
                "--stream_path", bin_dir,
                "--force_intra", "False",
                "--cuda", "True",
                "--worker", "1",
                "--verbose", "1",
                "--write_stream", "True",
                # Disable check_existing so we force a fresh attempt for this specific isolated run
                "--check_existing", "False" 
            ]

            # 4. Run it
            try:
                print(f"[RobustRunner] Running command...")
                subprocess.run(cmd, check=True)
                print(f"[RobustRunner] SUCCESS: {seq_name}")
            except subprocess.CalledProcessError:
                print(f"[RobustRunner] CRASH DETECTED: {seq_name} failed. Moving to next.")
            except Exception as e:
                print(f"[RobustRunner] ERROR: {e}")
            finally:
                if os.path.exists(temp_config_name):
                    os.remove(temp_config_name)

if __name__ == "__main__":
    main()