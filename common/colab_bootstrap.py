"""
common/colab_bootstrap.py
Reusable Colab session bootstrap: clone/pull repo, set up logging + auto-commit helpers.
"""
import os, subprocess, datetime, logging

def setup_repo(hw_folder, github_username, repo_name, token):
    """Clone repo if not present, else pull latest. Returns repo path."""
    repo_path = f"/content/{repo_name}"
    if not os.path.exists(repo_path):
        os.system(f"git clone https://{token}@github.com/{github_username}/{repo_name}.git {repo_path}")
    else:
        os.system(f"cd {repo_path} && git pull")

    os.system("git config --global user.email 'vedika.sumbli@sjsu.edu'")
    os.system("git config --global user.name 'vedika0806'")

    hw_path = os.path.join(repo_path, hw_folder)
    os.makedirs(hw_path, exist_ok=True)
    os.chdir(hw_path)
    print(f"Working directory: {os.getcwd()}")
    return hw_path

def get_logger(hw_path):
    """Attach a logger that appends to this HW's RUN_LOG.txt."""
    log_path = os.path.join(hw_path, "RUN_LOG.txt")
    logger = logging.getLogger(hw_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger

def git_commit_push(repo_path, message):
    """Stage everything, commit, push — call after every meaningful step."""
    subprocess.run(["git", "add", "."], cwd=repo_path)
    result = subprocess.run(["git", "commit", "-m", message], cwd=repo_path,
                             capture_output=True, text=True)
    print(result.stdout, result.stderr)
    subprocess.run(["git", "push"], cwd=repo_path)

def git_tag_push(repo_path, tag):
    subprocess.run(["git", "tag", tag], cwd=repo_path)
    subprocess.run(["git", "push", "origin", tag], cwd=repo_path)