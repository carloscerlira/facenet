# FaceNet
FaceNet model implemented by UNAM-IIMAS students as final project of Machine Learning class.  
You can find information about the implementation on paper.pdf  
In order to run the model, you will need to:
1. Download vggface2 (https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b) test and store it in datasets/train 
2. Download lfw (http://vis-www.cs.umass.edu/lfw/) and store it datasets/test 
3. Download iimas-model.pth (https://github.com/carloscerlira/facenet/releases/download/v1/iimas_model.pth) and put it inside logs
4. Run the script reproduce.py, from here you can change parameters such as epochs, batch size, etc.
5. You can choose to train a new model or use the petrained we obtained, if you choose to train, a graphics card with > 6GB of RAM is needed.
6. Once finished, you will find the results (roc curve, loss curve) inside logs/img  
