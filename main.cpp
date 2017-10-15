#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include <thread>

// includes for file_exists and files_in_directory functions
#ifndef __linux
#include <io.h> 
#define access _access_s
#else
#include <unistd.h>
#include <memory>
#endif
#define _DEBUG 0
#define POSITIVE_TRAINING_SET_PATH "dataset\\pos\\"
#define NEGATIVE_TRAINING_SET_PATH "dataset\\neg\\"
#define WINDOW_NAME "WINDOW"
#define TRAFFIC_VIDEO_FILE "V004.seq"
#define TRAINED_SVM "PD.yml"
#define	IMAGE_SIZE Size(64, 128) 
#define TERMCRITERIACOUNT 100

using namespace cv;
using namespace cv::ml;
using namespace std;

bool file_exists(const string &file);
void load_images(string directory, vector<Mat>& image_list);
vector<string> files_in_directory(string directory);

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size);
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
void test_it(const Size & size);

int main(int argc, char** argv)
{
	if (!file_exists(TRAINED_SVM)) {
		vector< Mat > pos_lst;
		vector< Mat > full_neg_lst;
		vector< Mat > neg_lst;
		vector< Mat > gradient_lst;
		vector< int > labels;

		cout << "Loading positive images..." << endl;
		load_images(POSITIVE_TRAINING_SET_PATH, pos_lst);
		labels.assign(pos_lst.size(), +1);
		cout << "Positive images Loaded Successfully!" << endl<<endl;

		cout << "Loading negative images..." << endl;
		load_images(NEGATIVE_TRAINING_SET_PATH, full_neg_lst);
		labels.insert(labels.end(), full_neg_lst.size(), -1);
		cout << "negative images Loaded Successfully!" << endl<<endl;


		cout << "Computing HOG features..." << endl;
		compute_hog(pos_lst, gradient_lst, IMAGE_SIZE);

		compute_hog(full_neg_lst, gradient_lst, IMAGE_SIZE);
		cout << "HOG featurescomputed successfully!" << endl<<endl;

		train_svm(gradient_lst, labels);
	}

	test_it(IMAGE_SIZE);
	return 0;
}

bool file_exists(const string &file)
{
	return access(file.c_str(), 0) == 0;
}

vector<string> files_in_directory(string directory)
{
	vector<string> files;
	char buf[256];
	string command;

#ifdef __linux__ 
	command = "ls " + directory;
	shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);

	char cwd[256];
	getcwd(cwd, sizeof(cwd));

	while (!feof(pipe.get()))
		if (fgets(buf, 256, pipe.get()) != NULL) {
			string file(cwd);
			file.append("/");
			file.append(buf);
			file.pop_back();
			files.push_back(file);
		}
#else
	command = "dir /b /s " + directory;
	FILE* pipe = NULL;

	if (pipe = _popen(command.c_str(), "rt"))
		while (!feof(pipe))
			if (fgets(buf, 256, pipe) != NULL) {
				string file(buf);
				file.pop_back();
				files.push_back(file);
			}
	_pclose(pipe);
#endif

	return files;
}

void load_images(string directory, vector<Mat>& image_list) {

	Mat img;
	vector<string> files;
	files = files_in_directory(directory);

	for (int i = 0; i < files.size(); ++i) {

		img = imread(files.at(i));
		if (img.empty())
			continue;

		if (img.cols >= 96 && img.rows >= 160)
			img = img(Rect(16, 16, 64, 128)); // Cut the 96 * 160 INRIA positive sample image to 64 * 128, that is, cut up and down the 16 pixels
#ifdef _DEBUG
		//imshow("image", img);
		//waitKey(10);
		//cout << img.size() << endl;
#endif
		
		image_list.push_back(img.clone());
	}
}

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);

	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	vector< Mat >::const_iterator itr = train_samples.begin();
	vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}

void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size)
{
	Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	srand((unsigned int)time(NULL));

	vector< Mat >::const_iterator img = full_neg_lst.begin();
	vector< Mat >::const_iterator end = full_neg_lst.end();
	for (; img != end; ++img)
	{
		box.x = rand() % (img->cols - size_x);
		box.y = rand() % (img->rows - size_y);
		Mat roi = (*img)(box);
		neg_lst.push_back(roi.clone());
#ifdef _DEBUG
		imshow("img", roi.clone());
		waitKey(10);
#endif
	}
}

// From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180° into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu

void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size)
{
	HOGDescriptor hog;
	hog.winSize = size;
	Mat gray;
	vector< Point > location;
	vector< float > descriptors;
	/*hog.blockSize.width = 5;
	hog.blockSize.height = 10;
	hog.cellSize.width = 1;
	hog.cellSize.height = 2;
	hog.blockStride.width = 5;
	hog.blockStride.height = 5;*/
	cout << hog.winSize.width << " " << hog.blockSize.width << " " << hog.blockSize.height << endl;
	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();

	for (; img != end; ++img)
	{
		cvtColor(*img, gray, COLOR_BGR2GRAY);

		hog.compute(gray, descriptors, Size(8, 8), Size(0, 0), location);

		gradient_lst.push_back(Mat(descriptors).clone());
#ifdef _DEBUG
		//imshow("gradient", get_hogdescriptor_visu(img->clone(), descriptors, size));
		//waitKey(10);
#endif
	}
}

void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels)
{
	//CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TERMCRITERIACOUNT, FLT_EPSILON);

	Ptr<SVM> svm = SVM::create();
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-3));
	svm->setGamma(0.0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

	clog << "Start training...";
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;

	svm->save(TRAINED_SVM);
}

void draw_locations(Mat&  img, const vector< Rect > & locations, const Scalar & color)
{
	//register Mat holding = img;
	if (!locations.empty())
	{
		
		register vector< Rect >::const_iterator loc = locations.begin();
		register vector< Rect >::const_iterator end = locations.end();

		for (; loc != end; ++loc)
		{
			rectangle(img, *loc, color, 2);
		//	Rect region_of_interest = Rect(0, 50, w, h - 50);
			//img = holding;

			
		}
	}
}

void test_it(const Size & size)
{
	int ct = 0, ct2 = 1;
	char key = 27;
	Mat img, draw1,draw2,draw;
	Ptr<SVM> svm;
	HOGDescriptor hog1,hog2;
	hog1.winSize = size;
	hog2.winSize = size;
	//VideoCapture video;
	vector< Rect > locations1, locations2, location_filtered1, location_filtered2,locations,locations_filtered;

	// Load the trained SVM.
	svm = StatModel::load<SVM>(TRAINED_SVM);
	// Set the trained svm to my_hog
	vector< float > hog_detector;
	//cout<<hog_detector.size()<<endl;
	get_svm_detector(svm, hog_detector);
	hog1.setSVMDetector(hog_detector);
	hog2.setSVMDetector(hog_detector);


	// Open the camera.
	/*video.open(TRAFFIC_VIDEO_FILE);
	if (!video.isOpened())
	{
		cerr << "Unable to open the device" << endl;
		exit(-1);
	}
*/
	int num_of_vehicles = 0;

	bool end_of_process = false;
	while (!end_of_process)
	{
	//	video >> img;
		stringstream filename;
		filename << "./V004seq/" << to_string(ct2++) << ".jpg";
		img = imread(filename.str());
		if (img.empty())
			break;

		
		
		//ROI
		int h = img.size().height;
		int w = img.size().width;
///////////////////////***Display width and Height of the image***//////////////////////////////////////////////
		
		/*cout << "Width : " << w;
		cout << "			Height: " << h << endl;

		cout << "Rows : " << img.rows;
		cout << "			columns: " << img.cols << endl;*/

//////////////////////////***Region of interest***//////////////////////////////////////////////////////////////
		//Cropping the upper useless pixels to decrease the size of image to be processed
		//This step will increase the speed
		Rect region_of_interest = Rect(0, 150, w, h-150);
		cout << region_of_interest << endl;
		Mat frame = img(region_of_interest); //region_of_interest = Rect(x, y, w, h);

/////////////////////////////////////////***Resizing***///////////////////////////////////////////////////////

		//resizing the orignal images
		resize(frame, frame, cv::Size(), 0.6, 0.6);
////////////////////////////////////////////////////////////////////////////////////////////
		locations1.clear();
		locations2.clear();
		locations.clear();

		h = frame.size().height;
		w = frame.size().width;
////////////////////////////////////***Dividing image into 2 halves***//////////////////////////////////
		//Dividing image into two halves
		Mat img1 = frame(Rect(0, 0, w / 2, h)); //region_of_interest = Rect(x, y, w, h);
		Mat img2 = frame(Rect(w / 2, 0, w / 2, h)); //region_of_interest = Rect(x, y, w, h);

/////////////////////////////////////////////////////////////////////////////////////////////////////////
		int64 t00 = cv::getTickCount();



//Thread1

		thread t1([&img1, &locations1, &hog1, &draw1]() {

			//calculate mutiscale HOG and draw locations
			hog1.detectMultiScale(img1, locations1, 0, Size(8, 8), Size(8, 8), 1.05, 2);
			draw1 = img1.clone();
			draw_locations(draw1, locations1, Scalar(0, 255, 0));
			//cout << "Thread 1: 1'm running " << endl;
		}

		);
//Thread2		
		thread t2([&img2, &locations2, &hog2, &draw2]() {

			//calculate mutiscale HOG and draw locations
			hog2.detectMultiScale(img2, locations2, 0, Size(8, 8), Size(8, 8), 1.05, 2);
			draw2 = img2.clone();
			draw_locations(draw2, locations2, Scalar(0, 255, 0));
		//	cout << "Thread 2: 1'm running " << endl;
		}

		);

		t1.join();
		t2.join();
////////////////////////////////////////////////////////////////////////////////////////////////

		//For No thread detection

		//cout << "threads completed" << endl;
		/*HOGDescriptor hog;
		hog.setSVMDetector(hog_detector);

		hog.detectMultiScale(frame, locations,0,Size(8,8),Size(8,8),1.05,2);
*/
		//draw = frame.clone();
		//draw_locations(draw, locations, Scalar(0, 255, 0));
		////cout << locations;

//////////////////////////////////////////////////////////////////////////////////////////////////////

		int64 t11 = cv::getTickCount();
		double secs = (t11 - t00) / cv::getTickFrequency();
		cout << "PED took " << secs << " seconds" << endl;
		
		//concatinate both images
		hconcat(draw1, draw2, draw); //Faster than the Adjust
		imshow(WINDOW_NAME, draw);
		waitKey(10);
		key = (char)waitKey(10);
		if (27 == key)
			end_of_process = true;

////////////////////////*** Writes detected windows to specfic folder***/////////////////////////////////////

		//size_t i, j;
		//for (i = 0; i < locations.size(); i++)
		//{
		//	Rect r = locations[i];
		//	for (j = 0; j <  locations.size(); j++)
		//		if (j != i && (r &  locations[j]) == r)
		//			break;
		//	if (j == locations.size())
		//		location_filtered.push_back(r);
		//}

		//for (i = 0; i < location_filtered.size(); i++)
		//{
		//	Rect r = location_filtered[i];
		//	if (r.x < 0)
		//		r.x = 0;
		//	if (r.y < 0)
		//		r.y = 0;
		//	if (r.x + r.width > frame.cols)
		//		r.width = frame.cols - r.x;
		//	if (r.y + r.height > frame.rows)
		//		r.height = frame.rows - r.y;
		//	Mat imgROI = frame(Rect(r.x, r.y, r.width, r.height));
		//	resize(imgROI, imgROI, Size(64, 128));
		//	stringstream filename;
		//	filename << "./Hard/v004fth" << to_string(ct++) << ".jpg";
		//	/*imshow(WINDOW_NAME, imgROI);
		//	key = (char)waitKey(10);
		//	if (27 == key)
		//		end_of_process = true;*/

		//	imwrite(filename.str(), imgROI);

		//}
////////////////////////////////////////////////////////////////////////////////////////////////////////////

	}
}
