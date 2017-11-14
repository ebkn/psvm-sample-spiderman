import psvm.*;

SVM model;
Histogram histogram;

int numBins = 256;

PImage testImage;
double testResult;

int []labels;
float [][]trainingData;
String []testFileNames;

float correctPercentage = 0.0;

void setup() {
  size(800, 860);
  textSize(32);
  
  // import moviews folder
  int folderCount = 0;
  
  java.io.File s1Folder = new java.io.File(dataPath("spider-man"));
  String []s1FileNames = s1Folder.list();
  folderCount += s1FileNames.length;
  
  java.io.File s2Folder = new java.io.File(dataPath("spider-man2"));
  String []s2FileNames = s2Folder.list();
  folderCount += s2FileNames.length;
  
  java.io.File s3Folder = new java.io.File(dataPath("spider-man3"));
  String []s3FileNames = s3Folder.list();
  folderCount += s3FileNames.length;
  
  java.io.File amFolder = new java.io.File(dataPath("amazing"));
  String []amFileNames = amFolder.list();
  folderCount += amFileNames.length;
  
  java.io.File am2Folder = new java.io.File(dataPath("amazing2"));
  String []am2FileNames = am2Folder.list();
  folderCount += am2FileNames.length;
  
  java.io.File homeFolder = new java.io.File(dataPath("homecoming"));
  String []homeFileNames = homeFolder.list();
  folderCount += homeFileNames.length;
  
  // initialize labels and training arrays
  labels = new int[folderCount];
  
  // one vector for each training image
  trainingData = new float[labels.length][numBins*3];
  
  // create the histogram object and tell it how many bins we want
  histogram = new Histogram();
  histogram.setNumBins(numBins);
  
  
  // build vectors
  PImage img;
  int startNum = 0;
  for (int i=0; i<s1FileNames.length; i++) {
    println("loading s1 : " + i);
    img = loadImage("spider-man/" + s1FileNames[i]);
    img.resize(800, 800);
    trainingData[startNum+i] = buildVector(img);
    labels[startNum+i] = 1;
  }
  
  startNum += s1FileNames.length;
  for (int i=0; i<s2FileNames.length; i++) {
    println("loading s2 : " + i);
    img = loadImage("spider-man2/" + s2FileNames[i]);
    img.resize(800, 800);
    trainingData[startNum+i] = buildVector(img);
    // train as spider-man
    labels[startNum+i] = 1;
  }
  
  startNum += s2FileNames.length;
  for (int i=0; i<s3FileNames.length; i++) {
    println("loading s3 : " + i);
    img = loadImage("spider-man3/" + s3FileNames[i]);
    img.resize(800, 800);
    trainingData[startNum+i] = buildVector(img);
    // train as spider-man
    labels[startNum+i] = 1;
  }
  
  startNum += s3FileNames.length;
  for (int i=0; i<amFileNames.length; i++) {
    println("loading am : " + i);
    img = loadImage("amazing/" + amFileNames[i]);
    img.resize(800, 800);
    trainingData[startNum+i] = buildVector(img);
    labels[startNum+i] = 4;
  }
  
  startNum += amFileNames.length;
  for (int i=0; i<am2FileNames.length; i++) {
    println("loading am2 : " + i);
    img = loadImage("amazing2/" + am2FileNames[i]);
    img.resize(800, 800);
    trainingData[startNum+i] = buildVector(img);
    // train as amazing
    labels[startNum+i] = 4;
  }
  
  startNum += am2FileNames.length;
  for (int i=0; i<homeFileNames.length; i++) {
    println("loading home : " + i);
    img = loadImage("homecoming/" + homeFileNames[i]);
    img.resize(800, 800);
    trainingData[startNum+i] = buildVector(img);
    labels[startNum+i] = 6;
  }
  
  
  // setup SVM model
  model = new SVM(this);
  SVMProblem problem = new SVMProblem();
  problem.setNumFeatures(numBins*3);
  problem.setSampleData(labels, trainingData);
  model.train(problem);
  
  // load in test images
  java.io.File testFolder = new java.io.File(dataPath("spider-test/"));
  testFileNames = testFolder.list();
  
  // run the evaluation
  evaluateResults();
  
  // load and test a new image
  loadNewTestImage();
}

// calculates the histogram for a PImage
// the results are used as the feature vector for SVM
float[] buildVector(PImage img) {
  histogram.setImage(img);
  histogram.calculateHistogram();
  histogram.scale(0, 0.33);
  
  return histogram.getRGB();
}

// load a new random image
void loadNewTestImage(){
  int imgNum = (int)random(0, testFileNames.length-1);
  testImage = loadImage("spider-test/" + testFileNames[imgNum]); 
  testImage.resize(800, 800);
  testResult = model.test(buildVector(testImage));
}

void evaluateResults(){
  int numCorrect = 0;
  PImage img;
  
  for (int i = 0; i < testFileNames.length; i++) {
    img = loadImage("spider-test/" + testFileNames[i]); 
    double r = model.test(buildVector(img));
    
    if(r == 1.0 && split(testFileNames[i], "-")[0].equals("spider-man")){
      numCorrect++;
    }
    // as spider-man
    if(r == 1.0 && split(testFileNames[i], "-")[0].equals("spider-man2")){
      numCorrect++;
    }
    // as spider-man
    if(r == 1.0 && split(testFileNames[i], "-")[0].equals("spider-man3")){
      numCorrect++;
    }
    if(r == 4.0 && split(testFileNames[i], "-")[0].equals("amazing")){
      numCorrect++;
    }
    // as amazing
    if(r == 4.0 && split(testFileNames[i], "-")[0].equals("amazing2")){
      numCorrect++;
    }
    if(r == 6.0 && split(testFileNames[i], "-")[0].equals("homecoming")){
      numCorrect++;
    }
  }
  
  correctPercentage = (float)numCorrect/testFileNames.length;
  println("Num Bins: " + numBins +
    " Percent Correct: " + correctPercentage);
}


void draw() {
  background(0);
  
  // display test image
  pushMatrix();
    scale(0.5);
    image(testImage, 0, 60, 800, 800);
  popMatrix();
  
  // testResult is set
  String message = "";
  if ((int)testResult == 1) {
    message = "spider man";
  } else if ((int)testResult == 2) {
    message = "spider man2";
  } else if ((int)testResult == 3) {
    message = "spider man3";
  } else if ((int)testResult == 4) {
    message = "amazing spider man";
  } else if ((int)testResult == 5) {
    message = "amazing spider man2";
  } else if ((int)testResult == 6) {
    message = "spider man homecoming";
  }
  
  // dispalay classification result
  fill(-1);
  text(message, 10, 25);
  
  // display correct percentage
  text("Percent classified correctly : " + 
    nf(correctPercentage * 100, 2, 2), 20, height -40);
  
  // display histogram
  stroke(-1);
  histogram.drawRGB(0, height/2, width, height/2);
}

void mousePressed() {
  loadNewTestImage();
}