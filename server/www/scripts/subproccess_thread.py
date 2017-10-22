import subprocess

def run_thread(argv=[""]):
    """
    Return a string that is the output from subprocess
    """

    # There is a link above on how to do this, but here's my attempt
    # I think this will work for your Python 2.6
    p = subprocess.Popen(argv, stdout=subprocess.PIPE)
    out, err = p.communicate()

    return out