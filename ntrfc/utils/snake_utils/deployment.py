import os
import shutil


def deploy(workflow_name, dest):
    """
    copies (part of) a workflow to a destination
    """
    import ntrfc
    assert os.path.isdir(dest), "destination is not a direcotry"
    assert len(os.listdir(dest)) == 0, "destination is not empty"
    src_dir = os.path.abspath(os.path.dirname(ntrfc.__file__))
    smk_dir = os.path.join(src_dir, "..", "workflows")
    wkf_dir = os.path.join(smk_dir, workflow_name)
    shutil.copytree(wkf_dir, dest, dirs_exist_ok=True)
    return
