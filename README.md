# Face_IntelVis
Face_IntelVis is a complete end to end training and prediction for face-recognition which is easy to aplly.
<br><br>
<hr>
<b>Description</b>  <br><br>

  The project has 4 folders: <br><br>

  - <b> CSV_LIST : </b> contains list faces and ID <br>
  - <b> DataFace : </b> Datasets<br>
  - <b> Embeded_Face :</b> Embeded of each face after face_recognition endcoding<br>
  - <b> Utils :</b> utils for project <br>

  
  <br><br> 

  <b>>>> USAGE: </b> <br>
 1. For training:<br><br>
  Put new image folder (folder name is person name) in <b>DataFace</b> and paste: 
 <pre> python train.py </pre> 
 Press Enter to complete training.

2. For prediction: <br><br>
'''
    Usage: 
    
    <pre> from utils.annModel import Model
    
    Model.predict(list_endcodings)
    
    Return list names prediction

    </pre> 


 