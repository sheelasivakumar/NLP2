from flask import Flask, render_template, redirect, request
import predict

app = Flask(__name__,static_url_path='/static')

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def marks():
    f = request.files['userfile']
    if f:
        # Check for allowed file types if necessary
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if f.filename.split('.')[-1].lower() not in allowed_extensions:
            return "Invalid file type. Please upload an image."

        path = "./static/" + f.filename
        f.save(path)                                                                                
        try:
            caption = predict.predict_captions(path)
            result_dic={
                'image': path,
                'caption':caption
            }
            return render_template("index.html", caption=result_dic)
        except Exception as e:
            print(e)
            return "Error occurred during caption prediction."
    else:
        return "No file uploaded."

if __name__ == "__main__":
    app.run(debug=True)
