import subprocess
def run_terminal_command(command, sudo_password=None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True,
        executable='/bin/bash', encoding='utf8')
    if sudo_password is not None:
        process.communicate(sudo_password + '\n')
    return process