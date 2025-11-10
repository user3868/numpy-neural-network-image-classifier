This project is about classifying and identifying 
monsters in the game Logres of Swords and Sorcery. 
Running main.py will start model training. The result 
will return 0 or 1 to represent whether the target 
monster exists. You can test the model for different 
images by modifying test_image_path The prediction results. 
At the same time, the running results also include 3 
learning curves as a reference for adjusting parameters. 
We can use screenshotsToImages.py to crop the screenshots 
into small grid images. Then manually mark the categories 
of the images by Add 1_ or 0_ in front of the file name. In 
this way, the collection of the data set is completed. 
After that, we need to put the images into labelImage 
to start machine learning.