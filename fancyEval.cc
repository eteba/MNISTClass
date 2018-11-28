#include <stdint.h>
#include <stdlib.h>

/*
 * ROOT script for evaluating the MNIST classifier
 * output. It randomly selects numbers from the inputs,
 * processes them with the network and plot both the image
 * and the network response.
 */
int fancyEval()
{
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetPalette(55, 0);	// kGreyScale
  gStyle->SetNumberContours(256);
  
  // Open input file with new data and prepare the data loading
  TFile* fin_images = new TFile("../idx2root/data/t10k-images-idx3-ubyte.root", "READ");
  TTree* tin_images = (TTree*)fin_images->Get("MNIST");
  
  TFile* fin_labels = new TFile("../idx2root/data/t10k-labels-idx1-ubyte.root", "READ");	// Not needed.
  TTree* tin_labels = (TTree*)fin_labels->Get("MNIST");
    
  std::vector<uint8_t>* img;
  tin_images->SetBranchAddress("images", &img);
  
  int nentries = tin_images->GetEntries();
  
  // Create the Reader object.
  TMVA::Reader* reader = new TMVA::Reader("V:Color:!Silent");
  
  // We have to add all the variables by hand...
  Float_t pixels[28*28];
  char varName[5];
  for(int i=0; i<28*28; i++)
  {
	snprintf(varName, 5, "p%i", i+1);			// "p1" ... "p784"
	reader->AddVariable(varName, &pixels[i]);	// pixels[0] ... pixels[784]
  }
  
  // Book the method and set the location of the weights
  reader->BookMVA("myDNN", "dataset/weights/MNISTClassOut_DNN.weights.xml");
  
  // PROBLEM: We don't know the order of appearance of the
  //		  classes in our weights file. We should figure
  //		  out the class index of each one.
  unsigned char clIndex[10];
  char className[2];
  for(int clname = 0; clname<10; clname++)
  {
	snprintf(className, 2, "%i", clname);
	clIndex[clname] = reader->DataInfo().GetClassInfo(className)->GetNumber();
  }
  
  // Randomly take images and plot the results
  TH2I* himages = new TH2I("image", "image", 28, 0, 1, 28, 0, 1);
  TH1D* houtput = new TH1D("out", "out", 10, -0.5, 9.5);
  
  himages->GetXaxis()->SetLabelSize(0);
  himages->GetXaxis()->SetLabelOffset(999);
  himages->GetXaxis()->SetNdivisions(000);
  himages->GetYaxis()->SetLabelSize(0);
  himages->GetYaxis()->SetLabelOffset(999);
  himages->GetYaxis()->SetNdivisions(000);
  
  houtput->SetFillColor(kBlue);
  houtput->GetXaxis()->SetLabelSize(0.08);
  houtput->GetXaxis()->SetLabelOffset(0.01);
  houtput->GetXaxis()->SetNdivisions(110);
  houtput->GetYaxis()->SetRangeUser(0.0, 1.0);
  houtput->GetYaxis()->SetLabelSize(0.07);
  houtput->GetYaxis()->SetLabelOffset(0.01);
  houtput->GetYaxis()->SetNdivisions(210);
  
  TCanvas* cv = new TCanvas("cv", "cv", 500, 700);
  std::vector<Float_t> output_vec;
  bool keepGoing = true;
  TRandom3* rndGen = new TRandom3(0);
  do
  {
	// Get random entry
	int entry = (int)(rndGen->Uniform(0.0, nentries-1) + 0.5);
	tin_images->GetEvent(entry);
	
	// Copy the image into our variables array
	for(int i=0; i<28*28; i++)
	  pixels[i] = img->at(i);
	
	// Fill the image histogram
	for(int i=0; i<28; i++)
	  for(int j=0; j<28; j++)
		himages->SetBinContent(j+1, 28-i, pixels[i*28+j]);
	
	// Get the network response and fill the histogram with it
	output_vec = reader->EvaluateMulticlass("myDNN");
	for(int clname=0; clname<10; clname++)
	{
	  houtput->SetBinContent(clname+1, output_vec.at(clIndex[clname]));
	}
	
	// Draw histograms
	cv->cd();
	
	TPad* pimages = new TPad("pimages", "pimages", 0, 0.3, 1, 1);
	pimages->SetBottomMargin(0.05);
	pimages->Draw();
	pimages->cd();
	himages->Draw("COL");
	
	cv->cd();
	
	TPad* poutput = new TPad("poutput", "poutput", 0, 0, 1, 0.3);
	poutput->SetBottomMargin(0.18);
	poutput->SetTopMargin(0.05);
	poutput->SetGridx();
	poutput->SetGridy();
	poutput->Draw();
	poutput->cd();
	houtput->Draw();
	
	cv->Modified();
	cv->Update();
	
	// Stop iterating
	int keepG = -1;
	do{
	  printf("Continue? 1 for Y, 0 for N...\t");
	  cin >> keepG;
	}while(keepG == -1);
	
	if(keepG == 0)
	  keepGoing = false;
  }while(keepGoing);
  
  delete cv;
  delete himages;
  delete houtput;
  delete reader;
  delete fin_images;

  return nentries;  
}
