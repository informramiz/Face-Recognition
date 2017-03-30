Name "FaceTracking"
OutFile "FaceTracking.exe"
InstallDir $PROGRAMFILES\FaceTracking
InstallDirRegKey HKLM "Software\NSIS_FaceTracking" "Install_Dir"
RequestExecutionLevel admin

Page components 
Page directory 
Page instfiles 

UninstPage uninstConfirm
UninstPage instfiles

Section "FaceTracking (required)"

	SectionIn RO
	
	SetOutPath $INSTDIR
	File "FaceTrackingCode.exe"
	#File "haarcascade_profileface.xml"
	#File "haarcascade_frontalface_alt.xml"
	File "training-files\*"
	#File "lbpcascade_frontalface.xml"
	File "opencv_calib3d243.dll"
	File "opencv_calib3d243d.dll"
	File "opencv_contrib243.dll"
	File "opencv_contrib243d.dll"
	File "opencv_core243.dll"
	File "opencv_core243d.dll"
	File "opencv_features2d243.dll"
	File "opencv_features2d243d.dll"
	File "opencv_flann243.dll"
	File "opencv_flann243d.dll"
	File "opencv_highgui243.dll"
	File "opencv_highgui243d.dll"
	File "opencv_imgproc243.dll"
	File "opencv_imgproc243d.dll"
	File "opencv_legacy243.dll"
	File "opencv_legacy243d.dll"
	File "opencv_ml243.dll"
	File "opencv_ml243d.dll"
	File "opencv_objdetect243.dll"
	File "opencv_objdetect243d.dll"
	File "opencv_video243.dll"
	File "opencv_video243d.dll"
	File "opencv_videostab243.dll"
	File "opencv_videostab243d.dll"
	
	;write installation path in reg
	WriteRegStr HKLM Software\NSIS_FaceTracking "Install_Dir" "$INSTDIR"
	
	;write uninstallation path
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceTracking" "DisplayName" "NSIS FaceTracking"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceTracking" "UninstallString" '"$INSTDIR\FaceTracking_uninstall.exe"'
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceTracking" "NoModify" 1
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceTracking" "NoRepair" 1
	WriteUninstaller "FaceTracking_uninstall.exe"

SectionEnd

; Optional section (can be disabled by the user)
Section "Start Menu Shortcuts"

  CreateDirectory "$SMPROGRAMS\FaceTracking"
  CreateShortCut "$SMPROGRAMS\FaceTracking\FaceTracking_Uninstall.lnk" "$INSTDIR\FaceTracking_uninstall.exe" "" "$INSTDIR\FaceTracking_uninstall.exe" 0
  CreateShortCut "$SMPROGRAMS\FaceTracking\FaceTracking.lnk" "$INSTDIR\FaceTrackingCode.exe" "" "$INSTDIR\FaceTrackingCode.exe" 0
  
SectionEnd

Section "Uninstall"
  
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceTracking"
  DeleteRegKey HKLM SOFTWARE\NSIS_FaceTracking

  ; Remove files and uninstaller
  Delete $INSTDIR\FaceTracking.nsi
  Delete $INSTDIR\uninstall.exe

  ; Remove shortcuts, if any
  Delete "$SMPROGRAMS\FaceTracking\*.*"

  ; Remove directories used
  RMDir "$SMPROGRAMS\FaceTracking"
  RMDir "$INSTDIR"

SectionEnd