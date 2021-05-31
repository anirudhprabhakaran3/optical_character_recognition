import os
from typing_extensions import final
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from model import run_style_transfer, save_results

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '':
        flash('No selected file!')
        return redirect(request.url)
    else:
        filename = secure_filename(file1.filename)
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file1.save(content_path)
    
    if file2.filename == '':
        flash('No selected file!')
        return redirect(request.url)
    else:
        filename = secure_filename(file2.filename)
        style_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file2.save(style_image)
    
    best, best_loss = run_style_transfer(content_path, style_image, num_iterations=100)
    final_image = save_results(best, content_path, style_image)

    final_image = "{}.png".format(final_image)

    return render_template('index.html', image=final_image)

