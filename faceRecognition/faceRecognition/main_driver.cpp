#include <direct.h>
#include "RFaceRecognizer.h"
using namespace std;

int Start( RFaceRecognizer & face_recognizer );
int Start( RFaceRecognizer & face_recognizer, std::string video_file_name );

char * GetCurrentWorkingDirectory( )
{
	char * buffer = NULL;
	if ( ( buffer = _getcwd( NULL , 0 ) ) == NULL )
	{
		std::cerr << "Unable to get current working directory" << std::endl;
		exit(1);
	}

	return buffer;
}

int main( )
{
	string face_cascade_name = "\\lbpcascade_frontalface.xml";
	string video_file_name = "E:\\test-videos\\face-tracking\\4.3gp";
	face_cascade_name = GetCurrentWorkingDirectory() + face_cascade_name;

	std::string training_file_path = "\\lbp-live.yml";
	training_file_path = GetCurrentWorkingDirectory() + training_file_path;

	std::cout << "Loading data...";

	RFaceRecognizer face_recognizer ( face_cascade_name , training_file_path );

	std::cout << std::endl;

	char ch;
	do
	{
		Start( face_recognizer , video_file_name);
		cvDestroyAllWindows( );

		std::cout << "Go to main menu (y/n)?:";
		cin >> ch;

	}while( ch != 'n' && ch != 'N' );

	return 0;
}

int Start( RFaceRecognizer & face_recognizer )
{
	int choice = 1;
	VideoCapture cap( 0 );

	if( !cap.isOpened() )
	{
		cout << "Could not Open Camera";
		return -1;
	}

	if( face_recognizer.IsTrained( ) )
	{
		cout << "Enter 1 to train and 2 to recognize: ";
		cin >> choice;
	}

	if(choice == 1)
	{
		int update_model = 0;
		if( face_recognizer.IsTrained( ) )
		{
			std::cout << "Model is already trained. Please select one of options give below." << std::endl;
			std::cout << "0. Learn new model" << std::endl;
			std::cout << "1. Update existing model" << std::endl;

			std::cout << "Enter your option number: ";
			cin >> update_model;
		}

		int label;
		cout << "Enter label of subject for training (integer): ";
		cin >> label;

		bool status = false;
		
		if( update_model )
		{
			status = face_recognizer.Update ( cap , label);
		}
		else
		{
			status = face_recognizer.learn( cap , label );
		}

		if ( status == false )
		{
			cerr << "training failed" << std::endl;
			return 1;
		}
		else
		{
			cout << "training succcessful" << std::endl;
		}
	}
	else if(choice == 2 )
	{
		if( !face_recognizer.IsTrained() )
		{
			std::cerr << "Algorithm is not trained. Please train it first" << std::endl;
			return -1;
		}

		face_recognizer.recognize( cap , true );
	}

	return 0;
}

int Start( RFaceRecognizer & face_recognizer, std::string video_file_name )
{
	int choice = 1;
	VideoCapture cap( video_file_name );

	if( !cap.isOpened() )
	{
		cout << "Could not Open Camera";
		return -1;
	}

	if( face_recognizer.IsTrained( ) )
	{
		cout << "Enter 1 to train and 2 to recognize: ";
		cin >> choice;
	}

	if(choice == 1)
	{
		int update_model = 0;
		if( face_recognizer.IsTrained( ) )
		{
			std::cout << "Model is already trained. Please select one of options give below." << std::endl;
			std::cout << "0. Learn new model" << std::endl;
			std::cout << "1. Update existing model" << std::endl;

			std::cout << "Enter your option number: ";
			cin >> update_model;
		}

		int label;
		cout << "Enter label of subject for training (integer): ";
		cin >> label;

		bool status = false;
		
		if( update_model )
		{
			status = face_recognizer.Update ( cap , label);
		}
		else
		{
			status = face_recognizer.learn( cap , label );
		}

		if ( status == false )
		{
			cerr << "training failed" << std::endl;
			return 1;
		}
		else
		{
			cout << "training succcessful" << std::endl;
		}
	}
	else if(choice == 2 )
	{
		if( !face_recognizer.IsTrained() )
		{
			std::cerr << "Algorithm is not trained. Please train it first" << std::endl;
			return -1;
		}

		face_recognizer.recognize( cap , true );
	}

	return 0;
}