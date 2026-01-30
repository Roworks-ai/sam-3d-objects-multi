# Prediction interface for Cog - SAM3D Multi-view
# https://cog.run/python

import subprocess
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import List

from cog import BasePredictor, Input  # noqa: cog is available at runtime

class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Setup SAM3D environment:
        - Step 0: Run hydra patching
        - Step 1: Authenticate with HuggingFace
        - Step 2: Download checkpoints if not present
        """
        self.sam3d_dir = Path(__file__).parent
        self.model_tag = "hf"

        # Step 0: Run SAM3D hydra patching
        print("Running SAM3D hydra patching...")
        patch_script = self.sam3d_dir / "patching" / "hydra"
        if patch_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(patch_script), "."],
                    cwd=str(self.sam3d_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"Hydra patching warning: {result.stderr}")
                else:
                    print("Hydra patching completed successfully")
            except Exception as e:
                print(f"Hydra patching failed: {e}, continuing anyway...")

        # Step 1: Authenticate with HuggingFace using token from .env
        print("Authenticating with HuggingFace...")
        env_file = self.sam3d_dir / ".env"
        hf_token = None

        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('HF_TOKEN='):
                        hf_token = line.strip().split('=', 1)[1].strip('"\'')
                        break

        if hf_token:
            try:
                result = subprocess.run(
                    ["huggingface-cli", "login", "--token", hf_token],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"HuggingFace auth warning: {result.stderr}")
                else:
                    print("HuggingFace authentication successful")
            except Exception as e:
                print(f"HuggingFace authentication failed: {e}, continuing anyway...")
        else:
            print("No HF_TOKEN found in .env file")

        # Step 2: Download SAM3D checkpoints if not present
        checkpoint_dir = self.sam3d_dir / "checkpoints" / self.model_tag
        if not checkpoint_dir.exists():
            print(f"Downloading SAM3D checkpoints (tag: {self.model_tag})...")
            download_dir = self.sam3d_dir / "checkpoints" / f"{self.model_tag}-download"

            try:
                # Download using huggingface-cli
                result = subprocess.run(
                    [
                        "huggingface-cli", "download",
                        "--repo-type", "model",
                        "--local-dir", str(download_dir),
                        "--max-workers", "1",
                        "facebook/sam-3d-objects"
                    ],
                    cwd=str(self.sam3d_dir),
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    raise Exception(f"Download failed: {result.stderr}")

                # Move checkpoints to final location
                src_checkpoints = download_dir / "checkpoints"
                if src_checkpoints.exists():
                    shutil.move(str(src_checkpoints), str(checkpoint_dir))

                # Cleanup download directory
                if download_dir.exists():
                    shutil.rmtree(download_dir)

                print(f"Checkpoints downloaded to {checkpoint_dir}")
            except Exception as e:
                raise Exception(f"Failed to download SAM3D checkpoints: {e}")
        else:
            print(f"SAM3D checkpoints already exist at {checkpoint_dir}")

        print("SAM3D setup completed successfully")

    def predict(
        self,
        input_archive: Path = Input(
            description="ZIP or TAR archive containing images and masks. Images: X.png, Masks: X_mask.png"
        ),
        mask_prompt: str = Input(
            description="Mask folder name. If empty, images and masks are in the same directory. If specified, images are in input/images/, masks in input/{mask_prompt}/",
            default=""
        ),
        image_names: str = Input(
            description="Comma-separated image names (without extension), e.g., 'image1,view_a' or '1,2'. If empty, use all available images",
            default=""
        ),
        seed: int = Input(
            description="Random seed for reproducibility",
            default=42,
            ge=0,
            le=2147483647
        ),
        stage1_steps: int = Input(
            description="Stage 1 inference steps",
            default=50,
            ge=10,
            le=100
        ),
        stage2_steps: int = Input(
            description="Stage 2 inference steps",
            default=25,
            ge=10,
            le=100
        ),
        decode_formats: str = Input(
            description="Output formats, comma-separated: 'gaussian', 'mesh', or 'gaussian,mesh'",
            default="gaussian,mesh"
        ),
        model_tag: str = Input(
            description="Model checkpoint tag",
            default="hf"
        ),
        output_dir: str = Input(
            description="Output directory path. If specified, results are copied here. If empty, results stay in visualization/",
            default=""
        ),
    ) -> List[Path]:
        """
        Run SAM3D 3D reconstruction using run_inference.py subprocess
        """
        # Extract archive to temp directory
        work_dir = Path(tempfile.mkdtemp())
        input_path = work_dir / "input"
        input_path.mkdir()

        archive_path = str(input_archive)
        print(f"Extracting archive: {archive_path}")

        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(input_path)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(input_path)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}. Use .zip, .tar, .tar.gz, or .tgz")

        # Check if files are in a subdirectory
        contents = list(input_path.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            input_path = contents[0]

        print(f"Extracted to: {input_path}")
        print(f"Contents: {list(input_path.iterdir())}")

        # Build command for run_inference.py
        cmd = [
            "python", "./run_inference.py",
            "--input_path", str(input_path),
            "--seed", str(seed),
            "--stage1_steps", str(stage1_steps),
            "--stage2_steps", str(stage2_steps),
            "--decode_formats", decode_formats,
            "--model_tag", model_tag,
        ]

        # Add optional parameters
        if mask_prompt:
            cmd.extend(["--mask_prompt", mask_prompt])

        if image_names:
            cmd.extend(["--image_names", image_names])

        print(f"Running SAM3D inference: {' '.join(cmd)}")

        # Run inference subprocess with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=str(self.sam3d_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)

        process.wait()

        if process.returncode != 0:
            raise Exception(f"SAM3D inference failed with return code {process.returncode}")

        # Collect output files from visualization directory
        # Output is saved to visualization/{dir_name}/ based on input
        output_paths = []
        viz_dir = self.sam3d_dir / "visualization"

        if viz_dir.exists():
            # Find the most recently created output directory
            output_dirs = sorted(
                [d for d in viz_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if output_dirs:
                latest_output = output_dirs[0]
                print(f"Collecting results from {latest_output}")

                # If output_dir is specified, copy files there
                if output_dir:
                    output_dir_path = Path(output_dir)
                    output_dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"Copying results to: {output_dir_path}")

                    for file in latest_output.glob("*"):
                        if file.is_file() and file.suffix in ['.glb', '.ply']:
                            dest_file = output_dir_path / file.name
                            shutil.copy2(file, dest_file)
                            output_paths.append(Path(dest_file))
                            print(f"  Copied: {file.name} -> {dest_file}")
                else:
                    # Return files from visualization directory
                    for file in latest_output.glob("*"):
                        if file.is_file() and file.suffix in ['.glb', '.ply']:
                            output_paths.append(Path(file))
                            print(f"  Found output: {file.name}")

        # Cleanup temp directory
        shutil.rmtree(work_dir, ignore_errors=True)

        if not output_paths:
            raise Exception("No output files generated from inference")

        return output_paths
