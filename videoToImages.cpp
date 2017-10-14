#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include<sstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

using namespace std;



int main(int argc, char** argv)
{
	cvNamedWindow("video", CV_WINDOW_AUTOSIZE);
	VideoCapture cap("V004.seq"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	int ct = 1;
	Mat frame;

	//std::stringstream filename;

	while (1)
	{
		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}
	//	cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
		cout << "Width : " << frame.size().width;
	//	cout << "			Height: " << frame.size().height << endl;
	//	Mat img = frame(Rect(0, 100, 320, 140)); //region_of_interest = Rect(x, y, w, h);
												 	if (!frame.data) break;
												 //	char file_name[100]="";
												 //sprintf_s(file_name, "%d.jpg", ct + 1);
												 stringstream filename;
												 filename << "./V004seq/" << to_string(ct) << ".jpg";
												 	imwrite(filename.str(), frame);
		imshow("video", frame);
		ct++;
		char c = cvWaitKey(33);
		if (c == 27) break;
	}
}


