# Real-Time-Hand-Sign-Detection-with-Tensor-Flow-Using-Keras-CNN
REAL TIME HAND SIGN DETECTION (USING KERAS CNN)
SHREERAM.S1, Utkarsh Sharma2, Bassar Patel3
Rajalakshmi Engineering College, Thandalam, Chennai-602105 1, IBM India Private Limited, Bengaluru, Karnataka 5600712 3
{200701234@rajalakshmi.edu.in1,sshreeram0511@gmail.com1 utkarsh@edunetfoundation.org 2, bassar@edunetfoundation.org  3}


















ABSTRACT
Communication is defined as the act of sharing or exchanging information, ideas or feelings. To establish communication between two people, both are required to have knowledge and understanding of a common language. But in the case of deaf and mute people, the means of communication are different. To overcome their barrier, one can build a model based on machine learning. A model can be trained to recognize different gestures of sign language and translate them into English. This will help a lot of people in communicating and conversing with deaf and mute people. A method proposed train a TensorFlow model to create a real-time Sign Language Recognition system. The system achieves a good level of accuracy even with a limited size dataset.
Keywords: Sign Language Recognition (SLR), Computer Vision, Machine Learning, Indian Sign Language.

INRODUCTION
Communication can be defined as the act of transferring information from one place, person, or group to another. It consists of three components: the speaker, the message that is being communicated, and the listener. It can be considered successful only when whatever message the speaker is trying to convey is received and understood by the listener. It can be divided into different categories as follows : formal and informal communication, oral (face-to-face and distance) and written communication, non-verbal, grapevine, feedback, and visual communication, and the active listening.

The formal communication (official communication) is steered through the channels that are pre-determined. The unofficial or grapevine communication is the spontaneous communication between individuals in one’s profession that does not have any formal protocol or structure.

The oral communication (face-to-face and distance) is the communication in which words are exchanged between people who are present in front or at a distance (with the help of technology including voice and video calls, webinars, etc.). The written communication is the communication in which letters, emails, notices, or any other written form is used for communicating. The non-verbal communication is the communication that uses gestures, facial expressions, body language, etc.




The feedback communication happens when a person gives feedback on some product or service provided by an individual or a company. The visual communication occurs when a person gets information from a visual source like televisions, social networking, or any other source. Active listening is when a person listens to and understands what the other individual is trying to convey so that the communication becomes more meaningful and effective.

Non-verbal communication helps deaf and dumb people to communicate amongst themselves and with others. Deaf is a disability that impairs a person's hearing ability and makes them incapable to hear while dumb is a disability that impairs the speaking ability and makes them incapable to speak. Not being able to speak or listen makes it difficult to establish communication with others. This is where sign languages come into the role, it enables a person to communicate without words. But a problem still exists, not many people possess the knowledge of sign language. Deaf and dumb may be able to communicate amongst themselves using sign languages but it is still difficult for them to communicate with people having normal hearing and vice-versa due to the lack of knowledge of sign languages. This issue can be resolved using a technology-driven solution. By using such a solution, one can easily translate the gestures of sign language into the commonly spoken language, English.

A lot of research has been done in this field and there is still a need for further re- search. For gesture translation, data gloves, motion capturing systems, or sensors have been used. Vision-based SLR systems have also been developed previously. The existing Indian Sign Language Recognition system was developed using machine learning algorithms with MATLAB. Authors have worked on single-handed and double- handed gestures.

They used two algorithms to train their system, K Nearest Neighbors Algorithm and Back Propagation Algorithm. Their system achieved 93-96% accuracy. Though being highly accurate, it is not a real-time SLR system. The objective of this paper is to develop a real-time SLR system using TensorFlow object detection API and train it using a dataset that will be created using a webcam.

The rest of this paper after the introduction is organized as follows. Section 2 presents the related work on the SLR system. Section 3 describes the data acquisition and generation. Section 4 focuses on the methodology of the developed system. Section 5 presents the experimental evaluation of the system, and finally, Section 6 concludes the paper with future work.


BLOCK DIAGRAM
 ![image](https://user-images.githubusercontent.com/100523801/202209640-b12f72df-925f-40bb-98aa-8343762bb794.png)

Fig 1.1 Block Diagram

TECHNOLOGIES
CONVOLUTIONAL NEURAL NETWORK (CNN)
Deep learning is a very significant subset of machine learning because of its high performance across various domains. Convolutional Neural Network (CNN) is a powerful image processing deep learning type often using in computer vision that comprises an image and video recognition along with a recommender system and natural language processing (NLP).
CNN uses a multilayer system consists of the input layer, output layer, and a hidden layer that comprises multiple convolutional layers, pooling layers, fully connected layers.
PLATFORM AND PROGRAMMING LANGUAGE
Jupyter notebooks basically provides an interactive computational environment for developing Python based Data Science applications. They are formerly known as ipython notebooks. The following are some of the features of Jupyter notebooks that makes it one of the best components of Python ML ecosystem −
•	Jupiter notebooks can illustrate the analysis process step by step by arranging the stuff like code, images, text, output etc. in a step by step manner.
•	It helps a data scientist to document the thought process while developing the analysis process.
•	One can also capture the result as the part of the notebook.
•	With the help of Jupyter notebooks, we can share our work with a peer also.
Python is a programming language that is preferred for programming due to its vast features, applicability, and simplicity. The Python programming language best fits machine learning due to its independent platform and its popularity in the programming community.
DATA SET
 ![image](https://user-images.githubusercontent.com/100523801/202209687-70c0f1ac-18d8-4185-b07b-9eb46287d760.png)

Fig 1.2- I Love You Sign - Dataset


![image](https://user-images.githubusercontent.com/100523801/202209732-a298ef05-e812-4899-a4ba-94032b66522b.png)


 
Fig 1.3- No  - Dataset


![image](https://user-images.githubusercontent.com/100523801/202209803-0e69e047-f359-4b23-bfe8-d8ee7de1633e.png)

Fig 1.4- Sorry - Dataset
 ![image](https://user-images.githubusercontent.com/100523801/202209835-a3ee4701-7b54-420c-b7a6-684ebf0ffad6.png)

Fig 1.5- Super – Dataset
 ![image](https://user-images.githubusercontent.com/100523801/202209891-6859cfd3-703e-44c0-95cf-7c252ece03d9.png)

Fig 1.6- Yes – Dataset



RESULTS
 ![image](https://user-images.githubusercontent.com/100523801/202209924-9ee9aed1-fb5c-4864-aa56-d54b4eb7eaaf.png)

Fig 1.7- No – Result
  ![image](https://user-images.githubusercontent.com/100523801/202209970-40862df5-ae10-4b90-a5e8-a3d73021686e.png)
![image](https://user-images.githubusercontent.com/100523801/202210050-f636f37e-92e9-46a6-975e-40cb807b9142.png)
Fig 1.8- Super – Result
![image](https://user-images.githubusercontent.com/100523801/202210011-5e1c12a8-3ea6-4aee-ae77-e9639fa62709.png)

  
Fig 1.9- Sorry – Result

 ![image](https://user-images.githubusercontent.com/100523801/202210129-dab12925-43bf-4825-840c-e34b57b82fab.png)

Fig 1.10- Yes – Result

 
Fig 1.11- I Love You – Result
CNN with Keras (Accuracy = 98%)
CONCLUSION
Nowadays, applications need several kinds of images as sources of information for elucidation and analysis. Several features are to be extracted to perform various applications. When an image is transformed from one form to another such as digitizing, scanning, and communicating, storing, etc. degradation occurs. Therefore, the output image must undertake a process called image enhancement, which contains of a group of methods that seek to develop the visual presence of an image. Image enhancement is fundamentally enlightening the interpretability or awareness of information in images for human listeners and providing better input for other automatic image processing systems. Image then undergoes feature extraction using various methods to make the image more readable by the computer. Sign language recognition system is a powerful tool to prepare an expert knowledge, edge detect and the combination of inaccurate information from different sources. The intend of convolution neural network is to get the appropriate classification. The IBM Internship 2022 had enhanced my technical skills on the domain of artificial intelligence and via this opportunity I was able to explore lot about the machine learning to acquire the domain knowledge.
FUTURE SCOPE

The proposed sign language recognition system used to recognize sign language letters can be further extended to recognize gestures facial expressions. Instead of displaying letter labels it will be more appropriate to display sentences as more appropriate translation of language. This also increases readability.  scope of different sign languages can be increased. More training data can be added to detect the letter with more accuracy. This project can further be extended to convert the signs to speech.
