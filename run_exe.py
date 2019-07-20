import subprocess
def run(exe_path):
    process = subprocess.Popen([exe_path])
    process.wait()