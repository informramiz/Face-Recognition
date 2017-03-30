Name "FaceRecognition"
OutFile "Face Recognition.exe"
InstallDir $PROGRAMFILES\FaceRecognition
InstallDirRegKey HKLM "Software\NSIS_FaceRecognition" "Install_Dir"
RequestExecutionLevel admin

Page components 
Page directory 
Page instfiles 

UninstPage uninstConfirm
UninstPage instfiles

Section "FaceRecognition (required)"

	SectionIn RO
	
	SetOutPath $INSTDIR
	File "FaceRecognition.exe"
	File "lbpcascade_frontalface.xml"
	File "*.dll"
	
	;write installation path in reg
	WriteRegStr HKLM Software\NSIS_FaceRecognition "Install_Dir" "$INSTDIR"
	
	;write uninstallation path
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceRecognition" "DisplayName" "NSIS FaceRecognition"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceRecognition" "UninstallString" '"$INSTDIR\FaceRecognition_uninstall.exe"'
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceRecognition" "NoModify" 1
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceRecognition" "NoRepair" 1
	WriteUninstaller "FaceRecognition_uninstall.exe"

SectionEnd

; Optional section (can be disabled by the user)
Section "Start Menu Shortcuts"

  CreateDirectory "$SMPROGRAMS\FaceRecognition"
  CreateShortCut "$SMPROGRAMS\FaceRecognition\FaceRecognition_Uninstall.lnk" "$INSTDIR\FaceRecognition_uninstall.exe" "" "$INSTDIR\FaceRecognition_uninstall.exe" 0
  CreateShortCut "$SMPROGRAMS\FaceRecognition\FaceRecognition.lnk" "$INSTDIR\FaceRecognition.exe" "" "$INSTDIR\FaceRecognition.exe" 0
  
SectionEnd

Section "Uninstall"
  
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceRecognition"
  DeleteRegKey HKLM SOFTWARE\NSIS_FaceRecognition

  ; Remove files and uninstaller
  Delete $INSTDIR\FaceRecognition.nsi
  Delete $INSTDIR\uninstall.exe

  ; Remove shortcuts, if any
  Delete "$SMPROGRAMS\FaceRecognition\*.*"
  Delete "$PROGRAMFILES\FaceRecognition\*";

  ; Remove directories used
  RMDir "$SMPROGRAMS\FaceRecognition"
  RMDir "$PROGRAMFILES\FaceRecognition"
  RMDir "$INSTDIR"

SectionEnd