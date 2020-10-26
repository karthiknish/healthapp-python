import numpy as np
from flask import Flask,render_template,url_for,flash,redirect,request,send_from_directory
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
tf.config.experimental.list_physical_devices('GPU')
app=Flask(__name__,template_folder='template')
app.config['SECRET_KEY']='f3cfe9ed8fae309f02079dbf'
dir_path=os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER='uploads'
STATIC_FOLDER='static'

pneumonia=load_model('pneumonia.h5')
malaria=load_model('malaria.h5')

def apimalaria(full_path):
    data=image.load_img(full_path,target_size=(50,50,3))
    data=np.expand_dims(data,axis=0)
    data=data*1.0/255
    predicted=malaria.predict(data)
    return predicted
def apipneumonia(full_path):
    data=image.load_img(full_path,target_size=(64,64,3))
    data=np.expand_dims(data,axis=0)
    data=data*1.0/255
    predicted=pneumonia.predict(data)
    return predicted

@app.route('/uploadmalaria',methods=['POST','GET'])
def upload_file_malaria():
    if request.method=='GET':
        return render_template('malaria.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices={0:'PARASITIC',1:'Uninficted',2:'Invasive carcinomar',3:'Normal'}
            result = apimalaria(full_name)
            predicted_class=np.asscalar(np.argmax(result,axis=1))
            accuracy=round(result[0][predicted_class]*100,2)
            label=indices[predicted_class]
            return render_template('predict_malaria.html',image_file_name=file.filename,label=label,accuracy=accuracy)
        except:
            flash('Please select the image first !!','danger')
            return redirect(url_for('malaria'))
@app.route('/uploadpneumonia',methods=['POST','GET'])
def upload_file_pneumonia():
    if request.method=='GET':
        return render_template('pneumonia.html')
    else:
        
        try:
            file=request.files['image']
            full_name=os.path.join(UPLOAD_FOLDER,file.filename)
            file.save(full_name)
            indices={0:'Normal',1:'Pneumonia'}
            result=apipneumonia(full_name)
            if(result>50):
                label=indices[1]
                accuracy=result
            else:
                label=indices[0]
                accuracy=100-result
            return  render_template('predict_pneumonia.html',image_file_name=file.filename,label=label,accuracy=accuracy)
        except:
            flash('Please select the image first !!','danger')
            return redirect(url_for('pneumonia'))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/cancer')
def cancer():
    return render_template('cancer.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/malaria')
def malaria():
    return render_template('malaria.html')
@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

def ValuePredictor(to_predict_list,size):
    to_predict=np.array(to_predict_list).reshape(1,size)
    if(size==8):
        loaded_model=joblib.load('diabetes')
        result=loaded_model.predict(to_predict)
    elif(size==30):
        loaded_model=joblib.load('cancer')
        result=loaded_model.predict(to_predict)
    elif(size==12):
        loaded_model=joblib.load('kidney')
        result=loaded_model.predict(to_predict)
    elif(size==10):
        loaded_model=joblib.load('liver')
        result=loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods=['GET','POST'])
def result():
    if request.method=='POST':
        to_predict_list=request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list=list(map(float,to_predict_list))
        if(len(to_predict_list)==30):
            result=ValuePredictor(to_predict_list,30)
        elif(len(to_predict_list)==8):
            result=ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==12):
            result=ValuePredictor(to_predict_list,12)
        elif(len(to_predict_list)==11):
            result=ValuePredictor(to_predict_list,11)
        elif(len(to_predict_list)==10):
            result=ValuePredictor(to_predict_list,10)

    if(int(result)==1):
        prediction='Sorry ! Suffering'
    else:
        prediction='Congrats ! you are healthy'
    return(render_template('result.html',prediction=prediction))

if __name__=='__main__':
    app.run(debug=True)