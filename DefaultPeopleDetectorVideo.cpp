#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[])
{
	cvNamedWindow("video", CV_WINDOW_AUTOSIZE);
	VideoCapture cap("V004.seq"); // open the default camera
//	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat img, frame;
	HOGDescriptor hog;

	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//namedWindow("video capture");
	while (true)
	{
		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}

		if (!frame.data) break;

		//	cv::resize(frame, img, cv::Size(), 0.5, 0.5);

		int h = frame.size().height;
		int w = frame.size().width;
		cout << "Width : " << w;
		cout << "			Height: " << h << endl;

		cout << "Rows : " << frame.rows;
		cout << "			columns: " << frame.cols << endl;
		Rect region_of_interest = Rect(0, 0, w, h);
		cout << region_of_interest << endl;
		Mat img = frame(region_of_interest); //region_of_interest = Rect(x, y, w, h);
		//Mat img = frame;
		vector<Rect> found, found_filtered;
		int64 t00 = cv::getTickCount();
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.02, 2);

		size_t i, j;
		for (i = 0; i<found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j<found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}
		for (i = 0; i<found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.06);
			r.height = cvRound(r.height*0.9);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
		}
		//cv::resize(img, img, cv::Size(), 2, 2);
		int64 t11 = cv::getTickCount();
		double secs = (t11 - t00) / cv::getTickFrequency();
		cout << "PED took " << secs << " seconds" << endl;
		imshow("video", img);
		if (waitKey(20) >= 0)
			break;
	}
	return 0;
}
