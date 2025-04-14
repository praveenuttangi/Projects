from flask import Flask, Response, render_template
import subprocess
import shlex
import time

app = Flask('pk')

def execute_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        line = process.stdout.readline()
        if not line:
            break
        yield line
    while True:
        err_line = process.stderr.readline()
        if not err_line:
            break
        yield err_line
    process.wait()


def generate_output(command):
    yield 'retry: 1000\n\n'
    for line in execute_command(command):
        yield f'data: {line}\n\n'

@app.route('/stream/<command>')
def stream(command):
    return Response(generate_output(command), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)