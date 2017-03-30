#include "RFaceRecognizer.h"
#include <fstream>

const double RFaceRecognizer::EYE_SX = 0.10;
const double RFaceRecognizer::EYE_SY = 0.19;
const double RFaceRecognizer::EYE_SW = 0.40;
const double RFaceRecognizer::EYE_SH = 0.36; 

const int RFaceRecognizer::DESIRED_IMAGE_WIDTH = 400;
const int RFaceRecognizer::DESIRED_IMAGE_HEIGHT = 400;
const int RFaceRecognizer::DESIRED_FACE_WIDTH = 240;
const int RFaceRecognizer::DESIRED_FACE_HEIGHT = 320;

const int RFaceRecognizer::noOfSubjects = 10;
const int RFaceRecognizer::noOfImages = 100;
const int RFaceRecognizer::testCol = 34;

const string RFaceRecognizer::leftEyeFile = "C:\\opencv_2.4.3\\data\\haarcascades\\haarcascade_mcs_lefteye.xml";
const string RFaceRecognizer::rightEyeFile = "C:\\opencv_2.4.3\\data\\haarcascades\\haarcascade_mcs_righteye.xml";

RFaceRecognizer::RFaceRecognizer( std::string & face_cascade_file_path, std::string & training_file_path )
	:face_cascade_file_path_( face_cascade_file_path ) ,
	training_file_path_( training_file_path ),
	is_trained_( false )
{
	LoadData();
}

void RFaceRecognizer::LoadData( )
{
	if( face_cascade_classifier_.load( face_cascade_file_path_ ) == false )
	{
		std::cerr << "LoadData()::Error:: Unable to load face_cascade training file: " << face_cascade_file_path_ << std::endl;
		exit( 1 );
	}

	face_recognizer_ = cv::createLBPHFaceRecognizer( );

	ifstream input_file( training_file_path_ , std::ios::in );

	if( input_file.is_open() )
	{
		input_file.close();
		is_trained_ = true;

		face_recognizer_->load( training_file_path_ );
	}
}

string RFaceRecognizer::num2str ( int value )
{
	stringstream stream;
	stream << value;

	string str;
	stream >> str;

	return str;
}

int RFaceRecognizer::str2num ( const string & str )
{
	stringstream stream;
	stream << str;

	int value;
	stream >> value;

	return value;
}

bool RFaceRecognizer::FaceDetector ( const Mat & image , vector<Rect> & faces )
{
	Mat grayImage ;
	if ( image.channels( ) == 3 )
		cvtColor ( image , grayImage , CV_BGR2GRAY );
	else if ( image.channels ( ) == 4 )
		cvtColor ( image , grayImage , CV_BGRA2GRAY );
	else
		grayImage = image;

	equalizeHist ( grayImage , grayImage );

	face_cascade_classifier_.detectMultiScale ( grayImage , faces , 1.1 , 3 , 0 | CV_HAAR_SCALE_IMAGE , Size( grayImage.cols/4 , grayImage.cols/4 ) );

	if ( faces.size( ) == 0 )
	{
		cout << "No face detected" << endl;
		return false;
	}

	return true;
}

bool RFaceRecognizer::FaceDetector ( const Mat & image , Mat & dest )
{
	Mat grayImage ;
	if ( image.channels( ) == 3 )
		cvtColor ( image , grayImage , CV_BGR2GRAY );
	else if ( image.channels ( ) == 4 )
		cvtColor ( image , grayImage , CV_BGRA2GRAY );
	else
		grayImage = image;

	equalizeHist ( grayImage , grayImage );
	vector<Rect> faces;

	face_cascade_classifier_.detectMultiScale ( grayImage , faces );

	if ( faces.size( ) == 0 )
	{
		cout << "No face detected" << endl;
		return false;
	}

	dest = grayImage ( faces[0] );
	return true;
}

bool RFaceRecognizer::FaceDetector ( const Mat & image , Mat & dest, Rect & faceReturn )
{
	Mat grayImage ;
	if ( image.channels( ) == 3 )
		cvtColor ( image , grayImage , CV_BGR2GRAY );
	else if ( image.channels ( ) == 4 )
		cvtColor ( image , grayImage , CV_BGRA2GRAY );
	else
		grayImage = image;

	equalizeHist ( grayImage , grayImage );

	vector<Rect> faces;
	face_cascade_classifier_.detectMultiScale ( grayImage , faces , 1.1 , 3 , 0 | CV_HAAR_SCALE_IMAGE , Size( grayImage.cols/4 , grayImage.cols/4 ) );

	if ( faces.size( ) == 0 )
	{
		//cout << "No face detected" << endl;
		return false;
	}

	faceReturn = faces[0];
	dest = grayImage ( faces[0] );
	return true;
}

bool RFaceRecognizer::readDataSet ( const string & dataSetPath , std::vector<Mat> & images_vector , std::vector<int> & labels_vector )
{
	Mat_<double> dataMatrix;
	bool isFirstTime = true;
	string path;

	for ( int subjectNo = 1 ; subjectNo <= noOfSubjects ; subjectNo++ )
	{
		for ( int imageNo = 1 ; imageNo <= noOfImages ; imageNo++ )
		{
			path = dataSetPath + "\\s" + num2str(subjectNo) + "\\" + num2str(imageNo) + ".pgm";
			Mat image = imread ( path , CV_LOAD_IMAGE_GRAYSCALE );
			
			if ( !image.data )
			{
				std::cerr << "unable to load image : " << path << std::endl;
				return false;
			}

			std::cout << "image : " << path << " read " << std::endl;

			if ( subjectNo == 8 && imageNo == 4 )
			{
				int a=1;
			}

			if ( image.rows > 400 || image.cols > 400 )
				resize ( image , image , Size( RFaceRecognizer::DESIRED_IMAGE_WIDTH , RFaceRecognizer::DESIRED_IMAGE_HEIGHT ) );

			Mat face;
			bool status = FaceDetector ( image , face );
			//cv::resize( face , face , cv::Size( 30 , 30 ) );
			if (  status == false )
			{
				std::cerr << "unable to detect face for image : " << path << std::endl;
				continue;
			}
			std::cout << "face detected for : " << path << std::endl;
			//resize ( face , face , Size ( RFaceRecognizer::DESIRED_FACE_WIDTH , DESIRED_FACE_HEIGHT ) );

			smooth ( face );
			std::cout << "image smoothed" << std::endl;

			string outputPath = "new\\s" + num2str(subjectNo) + "\\" + num2str(imageNo) + ".jpg";
			imwrite ( outputPath , face );
			std::cout << "image written" << std::endl;

			images_vector.push_back ( face );
			labels_vector.push_back ( subjectNo );
			std::cout << "image pushed" << std::endl;
		}
	}

	return true;
}

bool RFaceRecognizer::eyeDetector ( const Mat & face , Rect & eye , string eyeFileName )
{
	CascadeClassifier classifier;

	if ( classifier.load ( eyeFileName ) == 0 )
	{
		cout << "unable to load training data for eye classifier" << endl;
		return false;
	}

	vector<Rect> eyes;
	classifier.detectMultiScale ( face , eyes , 1.1 , 2 , 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT , Size ( 80 , 80 ) );

	if ( eyes.size( ) == 0 )
	{
		cout << "No eye detected" << endl;
		return false;
	}

	eye = Rect ( eyes[0] );

	return true;
}

bool RFaceRecognizer::eyeTranslator ( Mat & face , const Rect & leftEyeRect , const Rect & rightEyeRect )
{
	int leftX = cvRound ( face.cols * EYE_SX );
	int topY = cvRound ( face.rows * EYE_SY );
	int widthX = cvRound ( face.cols * EYE_SW );
	int heightY = cvRound ( face.rows * EYE_SH );
	int rightX  = cvRound ( face.cols * ( 1.0 - EYE_SX - EYE_SW ) );

	Point leftEye = Point ( -1 , -1 );
	leftEye.x = leftEyeRect.x + leftEyeRect.width / 2 + leftX;
	leftEye.y = leftEyeRect.y + leftEyeRect.height / 2 + topY;

	Point rightEye = Point ( -1 , -1 );
	rightEye.x = rightEyeRect.x + rightEyeRect.width / 2 + leftX  ;
	rightEye.y = rightEyeRect.y + rightEyeRect.height / 2 + topY;

	Point2f eyesCenter;
	eyesCenter.x = (leftEye.x + rightEye.x) * 0.5f;
	eyesCenter.y = (leftEye.y + rightEye.y) * 0.5f;

	// Get the angle between the 2 eyes.
	double dy = (rightEye.y - leftEye.y);
	double dx = (rightEye.x - leftEye.x);
	double len = sqrt(dx*dx + dy*dy);

	// Convert Radians to Degrees.
	double angle = atan2(dy, dx) * 180.0/CV_PI;

	// Hand measurements shown that the left eye center should 
	// ideally be roughly at (0.16, 0.14) of a scaled face image.
	const double DESIRED_LEFT_EYE_X = 0.16;
	const double DESIRED_RIGHT_EYE_X = 1.0f - 0.16f;
	const double DESIRED_LEFT_EYE_Y = 0.14;
	const double DESIRED_RIGHT_EYE_Y = 1.0f - 0.14f;

	// Get the amount we need to scale the image to be the desired
	// fixed size we want.
	const int DESIRED_FACE_WIDTH = 100;
	const int DESIRED_FACE_HEIGHT = 100;
	double desiredLen = (DESIRED_RIGHT_EYE_X - 0.16);
	double scale = desiredLen * DESIRED_FACE_WIDTH / len;

	// Get the transformation matrix for the desired angle & size.
	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
	// Shift the center of the eyes to be the desired center.
	double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
	double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;
	rot_mat.at<double>(0, 2) += ex;
	rot_mat.at<double>(1, 2) += ey;
	// Transform the face image to the desired angle & size & 
	// position! Also clear the transformed image background to a 
	// default grey.
	Mat warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH , CV_8U, Scalar(128));
	warpAffine( face , warped, rot_mat, warped.size() );

	face = warped;

		int w = face.cols;
	int h = face.rows;
	Mat wholeFace;
	equalizeHist ( face , wholeFace );

	int midX = w / 2;
	Mat leftSide = face ( Rect ( 0 , 0 , midX , h ) );
	Mat rightSide = face ( Rect ( midX , 0 , w-midX , h ) );
	equalizeHist ( leftSide , leftSide );
	equalizeHist ( rightSide , rightSide );

	for (int y = 0; y < h; y++) 
	{
		for (int x = 0; x < w ; x++) 
		{
			int v;
			if ( x < w / 4 ) 
			{
				// Left 25%: just use the left face.
				v = leftSide.at<uchar>(y,x);
			}
			else if ( x < w * 2 / 4 ) 
			{
				// Mid-left 25%: blend the left face & whole face.
				int lv = leftSide.at<uchar>(y,x);
				int wv = wholeFace.at<uchar>(y,x);
				// Blend more of the whole face as it moves
				// further right along the face.
				float f = (x - w*1/4) / (float)(w/4);
				v = cvRound((1.0f - f) * lv + (f) * wv);
			}
			else if ( x < w * 3 / 4 ) 
			{
				// Mid-right 25%: blend right face & whole face.
				int rv = rightSide.at<uchar>(y,x-midX);
				int wv = wholeFace.at<uchar>(y,x);
				// Blend more of the right-side face as it moves
				// further right along the face.
				float f = (x - w * 2 / 4 ) / (float)( w / 4 );
				v = cvRound( ( 1.0f - f ) * wv + ( f ) * rv );
			}
			else 
			{
				// Right 25%: just use the right face.
				v = rightSide.at<uchar>(y,x-midX);
			}

			face.at<uchar>(y,x) = v;
		}// end x loop
	}//end y loop

	Mat filtered = Mat( face.size(), CV_8U);
	bilateralFilter( face , filtered , 0, 20.0, 2.0);

	// Draw a black-filled ellipse in the middle of the image.
// First we initialize the mask image to white (255).
	Mat mask = Mat( face.size( ) , CV_8UC1, Scalar( 255 ) );
	double dw = DESIRED_FACE_WIDTH;
	double dh = DESIRED_FACE_HEIGHT;

	Point faceCenter = Point( cvRound(dw * 0.5), cvRound( dh * 0.4 ) );
	Size size = Size( cvRound(dw * 0.5), cvRound(dh * 0.8) );
	ellipse( mask, faceCenter, size, 0, 0, 360, Scalar(0),  CV_FILLED );

	// Apply the elliptical mask on the face, to remove corners.
	// Sets corners to gray, without touching the inner face.
	filtered.setTo(Scalar(128), mask);
	face = filtered;

	return true;
}

void RFaceRecognizer::smooth ( Mat & face )
{
	int w = face.cols;
	int h = face.rows;
	Mat wholeFace;
	equalizeHist ( face , wholeFace );

	int midX = w / 2;
	Mat leftSide = face ( Rect ( 0 , 0 , midX , h ) );
	Mat rightSide = face ( Rect ( midX , 0 , w-midX , h ) );
	equalizeHist ( leftSide , leftSide );
	equalizeHist ( rightSide , rightSide );

	for (int y = 0; y < h; y++) 
	{
		for (int x = 0; x < w ; x++) 
		{
			int v;
			if ( x < w / 4 ) 
			{
				// Left 25%: just use the left face.
				v = leftSide.at<uchar>(y,x);
			}
			else if ( x < w * 2 / 4 ) 
			{
				// Mid-left 25%: blend the left face & whole face.
				int lv = leftSide.at<uchar>(y,x);
				int wv = wholeFace.at<uchar>(y,x);
				// Blend more of the whole face as it moves
				// further right along the face.
				float f = (x - w*1/4) / (float)(w/4);
				v = cvRound((1.0f - f) * lv + (f) * wv);
			}
			else if ( x < w * 3 / 4 ) 
			{
				// Mid-right 25%: blend right face & whole face.
				int rv = rightSide.at<uchar>(y,x-midX);
				int wv = wholeFace.at<uchar>(y,x);
				// Blend more of the right-side face as it moves
				// further right along the face.
				float f = (x - w * 2 / 4 ) / (float)( w / 4 );
				v = cvRound( ( 1.0f - f ) * wv + ( f ) * rv );
			}
			else 
			{
				// Right 25%: just use the right face.
				v = rightSide.at<uchar>(y,x-midX);
			}

			face.at<uchar>(y,x) = v;
		}// end x loop
	}//end y loop

	Mat filtered = Mat( face.size(), CV_8U);
	bilateralFilter( face , filtered , 0, 20.0, 2.0);
	face = filtered;

	Mat mask = Mat( face.size( ) , CV_8UC1, Scalar( 255 ) );
	double dw = face.cols - ( face.cols / 100 * 15 );
	double dh = face.rows - ( face.rows / 100 * 10 );

	Point faceCenter = Point( cvRound(dw * 0.5), cvRound( dh * 0.4 ) );
	Size size = Size( cvRound(dw * 0.5), cvRound(dh * 0.8) );
	ellipse( mask, faceCenter, size, 0, 0, 360, Scalar(0),  CV_FILLED );

	// Apply the elliptical mask on the face, to remove corners.
	// Sets corners to gray, without touching the inner face.
	filtered.setTo(Scalar(128), mask);
	face = filtered; 
}

bool RFaceRecognizer::readDataSet ( VideoCapture & cap , std::vector<Mat> & images_vector , std::vector<int> & labels_vector , int label)
{
	Mat_<double> dataMatrix;
	bool isFirstTime = true;
	string path;

	int subjectNo = 1;
	namedWindow("Training",1);

	for ( int imageNo = 1 ; imageNo <= noOfImages ; imageNo++ )
	{
		//path = dataSetPath + "\\s" + num2str(subjectNo) + "\\" + num2str(imageNo) + ".pgm";
		Mat image, original;
		Rect faceRect;

		cap >> original;
			
		if ( !original.data )
		{
			std::cerr << "unable to load image : " << path << std::endl;
			return false;
		}

		//original = original.t();
		imshow("Training", original);
		
		/*if ( original.rows > 400 || original.cols > 400 )
			resize ( original , original , Size( RFaceRecognizer::DESIRED_IMAGE_WIDTH , RFaceRecognizer::DESIRED_IMAGE_HEIGHT ) );*/

		Mat face;
		bool status = FaceDetector ( original , face , faceRect);
		//cv::resize( face , face , cv::Size( 30 , 30 ) );

		if (  status == false )
		{
			//std::cerr << "unable to detect face for image : " << path << std::endl;
			imageNo--;

			if(waitKey(30) >= 0)
			{
				break;
			}

			continue;
		}
		//resize ( face , face , Size ( RFaceRecognizer::DESIRED_FACE_WIDTH , DESIRED_FACE_HEIGHT ) );

		smooth ( face );
		
		rectangle(original, faceRect, CV_RGB(0, 255,0), 1);
		int pos_x = max(faceRect.tl().x - 10, 0);
        int pos_y = max(faceRect.tl().y - 10, 0);

		if( imageNo == 100 )
		{
			putText(original, "Training algorithm...", Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
		}
		else
		{
			putText(original, "Face Detected", Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
		}

		imshow("Training", original);
		
		images_vector.push_back ( face );
		labels_vector.push_back ( label );

		if(waitKey(30) >= 0)
		{
			break;
		}
	}


	return true;
}

bool RFaceRecognizer::learn ( VideoCapture & cap , int label )
{
	std::vector<cv::Mat> images_vector;
	std::vector<int> labels_vector;

	bool status = readDataSet ( cap , images_vector , labels_vector, label );
	
	if ( status == false )
	{
		std::cerr << "Learn()::Error::reading training images failed" << std::endl;
		return false;
	}

	if ( images_vector.size( ) <= 1 )
	{
		std::cerr << "Learn()::Error::minimum 2 images are required for LBP FaceRecognizer" << std::endl;
		return false;
	}

	face_recognizer_->train( images_vector , labels_vector );
	face_recognizer_->save( training_file_path_ );

	is_trained_ = true;
	return true;
}

bool RFaceRecognizer::Update ( VideoCapture & cap , int label )
{
	std::vector<cv::Mat> images_vector;
	std::vector<int> labels_vector;

	bool status = readDataSet ( cap , images_vector , labels_vector, label );
	
	if ( status == false )
	{
		std::cerr << "Learn()::Error::reading training images failed" << std::endl;
		return false;
	}

	if ( images_vector.size( ) <= 1 )
	{
		std::cerr << "Learn()::Error::minimum 2 images are required for LBP FaceRecognizer" << std::endl;
		return false;
	}

	if( is_trained_ )
	{
		face_recognizer_->update( images_vector , labels_vector );
	}
	else
	{
		face_recognizer_->train( images_vector , labels_vector );
	}

	face_recognizer_->save( training_file_path_ );
	return true;
}

bool RFaceRecognizer::learn ( const std::string & dataSetPath , const std::string & outputFilePath  )
{
	std::vector<cv::Mat> images_vector;
	std::vector<int> labels_vector;

	std::cout << "reading dataset" << std::endl;
	bool status = readDataSet ( dataSetPath , images_vector , labels_vector );

	if ( status == false )
	{
		std::cout << "reading dataset failed" << std::endl;
		return false;
	}

	if ( images_vector.size( ) <= 1 )
	{
		std::cerr << "minimum 2 images are required for LBP FaceRecognizer" << std::endl;
		return false;
	}

	cv::Ptr<cv::FaceRecognizer> faceRecognizer = cv::createLBPHFaceRecognizer( );
	faceRecognizer->train( images_vector , labels_vector );
	faceRecognizer->save( outputFilePath );

	return true;
}

bool RFaceRecognizer::recognize ( const cv::Mat & testImage , int & predictedLabel , double & confidence , cv::Rect & fRect)
{
	if ( !testImage.data )
	{
		std::cerr << "empty test image" << std::endl;
		return false;
	}

	//cv::resize ( testImage , testImage , cv::Size ( DESIRED_IMAGE_WIDTH , DESIRED_IMAGE_HEIGHT ) );
	cv::Mat face;
	bool status = FaceDetector( testImage , face , fRect);

	if ( status == false )
	{
		//std::cerr << "Recognize()::Error:: Unable to detect face from test image" << std::endl;
		return false;
	}

	smooth( face );

	//face_recognizer_->set("threshold", 10.0);
	face_recognizer_->predict( face , predictedLabel , confidence );

	return true;
}

bool RFaceRecognizer::recognize ( const cv::Mat & testImage , vector<Rect> & faces, vector<int> & labels, vector<double> & confidences )
{
	if ( !testImage.data )
	{
		std::cerr << "empty test image" << std::endl;
		return false;
	}

	//cv::resize ( testImage , testImage , cv::Size ( DESIRED_IMAGE_WIDTH , DESIRED_IMAGE_HEIGHT ) );
	bool status = FaceDetector( testImage , faces);

	if ( status == false )
	{
		//std::cerr << "Recognize()::Error:: Unable to detect face from test image" << std::endl;
		return false;
	}

	labels.resize( faces.size( ) );
	confidences.resize( faces.size( ) );

	cv::Mat face;
	for( int i = 0; i < faces.size( ) ; ++i )
	{
		face = testImage( faces[i] );
		smooth( face );

		//face_recognizer_->set("threshold", 10.0);
		face_recognizer_->predict( face , labels[i] , confidences[i] );
	}

	return true;
}

bool RFaceRecognizer::recognize ( const std::string & testImagePath , int & predictedLabel , double & confidence )
{
	Mat testImage = cv::imread ( testImagePath , CV_LOAD_IMAGE_GRAYSCALE );
	if ( !testImage.data )
	{
		std::cerr << "empty test image" << std::endl;
		return false;
	}

	//cv::resize ( testImage , testImage , cv::Size ( DESIRED_IMAGE_WIDTH , DESIRED_IMAGE_HEIGHT ) );
	cv::Mat face;
	cv::Rect fRect;
	bool status = FaceDetector( testImage , face , fRect);

	if ( status == false )
	{
		std::cerr << "Recognize()::Error:: Unable to detect face from test image" << std::endl;
		return false;
	}

	smooth( face );

	//face_recognizer_->set("threshold", 10.0);
	face_recognizer_->predict( face , predictedLabel , confidence );

	return true;
}

bool RFaceRecognizer::recognize( cv::VideoCapture & cap , bool recognize_multiple_faces )
{
	if( recognize_multiple_faces )
	{
		return recognizeMultipleFaces( cap );
	}
	else
	{
		return recognizeSingleFace( cap );
	}
}

bool RFaceRecognizer::recognizeSingleFace( cv::VideoCapture & cap )
{
	int predictedLabel = -1;
	double confidence = 0.0;

	bool status;
	cv::Mat frame;
	cv::namedWindow( "faces" , 1 );

	for( int i = 0 ; i < 1000 ; i++ )
	{
		cap >> frame;
	
		if ( !frame.data )
		{
			return false;
		}

		//frame = frame.t();
		imshow( "faces" , frame );

		cv::Rect rect;
		status = recognize ( frame , predictedLabel , confidence , rect );

		cv::rectangle( frame , rect, CV_RGB(0, 255,0), 1);

		int pos_x = max(rect.tl().x - 10, 0);
        int pos_y = max(rect.tl().y - 10, 0);

		std::string text = std::string( "Label: " ) + num2str( predictedLabel ) + std::string( ", Confidence: " ) + num2str( confidence );
		putText( frame , text , Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

		cv::imshow( "faces" , frame );

		if( cv::waitKey(30) >= 0 )
		{
			break;
		}
	}

	return true;
}

bool RFaceRecognizer::recognizeMultipleFaces( cv::VideoCapture & cap )
{
	bool status;
	cv::Mat frame;
	cv::namedWindow( "faces" , 1 );

	for( int i = 0 ; i < 1000 ; i++ )
	{
		cap >> frame;
	
		if ( !frame.data )
		{
			return false;
		}

		frame = frame.t();
		imshow( "faces" , frame );

		vector<Rect> faces;
		vector<int> labels;
		vector<double> confidences;

		status = recognize ( frame , faces , labels , confidences );

		for( int i = 0; i < faces.size( ) ; ++i )
		{
			cv::rectangle( frame , faces[i] , CV_RGB(0, 255,0), 1);

			int pos_x = max(faces[i].tl().x - 10, 0);
			int pos_y = max(faces[i].tl().y - 10, 0);

			std::string text = std::string( "Label: " ) + num2str( labels[i] ) + std::string( ", Confidence: " ) + num2str( confidences[i] );
			putText( frame , text , Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
		}

		cv::imshow( "faces" , frame );

		if( cv::waitKey(30) >= 0 )
		{
			break;
		}
	}

	return true;
}

bool RFaceRecognizer::IsTrained( )
{
	return is_trained_;
}