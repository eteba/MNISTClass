#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "TFile.h"
#include "TString.h"
#include "TTree.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Types.h"

/*
 * inputs: 
 * 	 argv[1]: .root images file
 * 	 argv[2]: .root labels file
 * 
 * Compilation:
 * 	 g++ MNISTClass.cc -o MNISTClass -I/usr/lib64/root/6.14/include -L/usr/lib64/root/6.14/lib64 -lCore -lRIO -lTree -lTMVA
 */
int main(int argc, char **argv)
{
  if(argc != 3)
  {
	printf("Usage: ./MNISTClass images.root labels.root.\n");
	return 0;
  }
  
  // Open input files and trees.
  TFile* fin_images = new TFile(argv[1], "READ");
  TFile* fin_labels = new TFile(argv[2], "READ");
  
  TTree* tin_images = (TTree*)fin_images->Get("MNIST");
  TTree* tin_labels = (TTree*)fin_labels->Get("MNIST");
  
  int nentries = tin_images->GetEntries();
  if(nentries != tin_labels->GetEntries())
  {
	fprintf(stderr, "Error: images and labels trees have a different number of items.\n");
	return 0;
  }
  
  // Create output file.
  TFile* fout = new TFile("output/outClass.root", "RECREATE");
  
  // Create the factory object and the data loader.
  TMVA::Factory* factory = new TMVA::Factory("MNISTClassOut", fout,
											 "V:!Silent:Color:DrawProgressBar:AnalysisType=Multiclass");
  
  TMVA::DataLoader* dataloader = new TMVA::DataLoader("dataset");
  
  // Variables:
  //	28*28 input neurons
  //	0 spectator variables
  //	0 target variables
  // -> AddEvent must be fed with a (28*28 + 0 + 0) elements vector.
  char varName[5];
  for(int i=1; i<=28*28; i++)
  {
	snprintf(varName, 5, "p%i", i);
	dataloader->AddVariable(varName, 'I');
  }
  
  // We create the 10 different classes and assign the data to each one.
  // 1 of each 5 events is chosen as test events.
  // The input variables must be stored in a vector. Exploiting
  // the way we stored the images, we can use those vectors.  
  std::vector<uint8_t> *img;
  int label;
  
  tin_images->SetBranchAddress("images", &img);
  tin_labels->SetBranchAddress("labels", &label);
  
  printf("Processing input entries.\n");
  int counter = 0;
  for(int n=0; n<nentries; n++)
  {
	if((n+1)%1000 == 0)
	  printf("Entry %i of %i\n", n+1, nentries);
	
	tin_images->GetEvent(n);
	tin_labels->GetEvent(n);
	counter++;
	
	// TMVA::DataLoader::AddEvent requires the vector to be double.
	const std::vector<double> img_double(img->begin(), img->end());
	
	// Store the label in a string so we can use it to create the class.
	char str_label[2];
	snprintf(str_label, 2, "%i", label);
	
	// Testing set.
	if(counter % 5 == 0)
	{
	  dataloader->AddEvent(str_label, TMVA::Types::kTesting, img_double, 1.0);	// weight=1.0
	}
	// Training set.
	else
	{
	  dataloader->AddEvent(str_label, TMVA::Types::kTraining, img_double, 1.0);
	}
  }
  printf("Events added.\n");
  dataloader->PrepareTrainingAndTestTree("", "");
  printf("SetInputTreesFromEventAssignTrees() done.\n");
  
  // Prepare the DNN.
  TString networkLayout("Layout=TANH|50,TANH|20,RELU|20,LINEAR");
  TString trainingStrategy1("BatchSize=100,ConvergenceSteps=10,DropConfig=0.0,LearningRate=1e-1,Momentum=0.5,Regularization=NONE,Multithreading=T,TestRepetitions=10");
  TString trainingStrategy2("BatchSize=300,ConvergenceSteps=20,DropConfig=0.0+0.5+0.5+0.0,LearningRate=1e-2,Momentum=0.25,Regularization=NONE,Multithreading=T,TestRepetitions=10");
  TString trainingStrategy3("BatchSize=500,ConvergenceSteps=50,DropConfig=0.0+0.25+0.25+0.0,LearningRate=1e-3,Momentum=0.0,Regularization=NONE,Multithreading=T,TestRepetitions=10");
  TString trainingStrategy("TrainingStrategy=");
  trainingStrategy += trainingStrategy1 + "|" + trainingStrategy2 + "|" + trainingStrategy3;
  TString DNNOptions("!H:V:Architecture=CPU:ErrorStrategy=CROSSENTROPY:WeightInitialization=XAVIERUNIFORM");
  DNNOptions.Append(":");
  DNNOptions.Append(networkLayout);
  DNNOptions.Append(":");
  DNNOptions.Append(trainingStrategy);
  
  // Book it in the factory object.
  factory->BookMethod(dataloader, TMVA::Types::kDNN, "DNN", DNNOptions);
  
  printf("Booked.\n");
  
  // Train the network.
  factory->TrainAllMethods();
  
  printf("Trained.\n");
  
  // Evaluate with the test set.
  factory->TestAllMethods();
  
  // Finish.
  fout->Close();
  
  delete factory;
  delete dataloader;
  
  return 0;
}
