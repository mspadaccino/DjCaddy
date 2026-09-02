; L'installer di Windows, per Inno Setup 6.
;
;     iscc packaging\djcaddy.iss
;
; Prende quello che PyInstaller ha lasciato in dist\DjCaddy\ (onedir, non
; onefile: 3-4 GB da riestrarre a ogni avvio non sono un'app) e ne fa un
; unico setup in dist\.
;
; Su Windows essentia non ha wheel: il bundle è completo di tutto tranne le
; funzioni che ne dipendono — Tag Maker e la COSTRUZIONE della mappa. Tutto
; il resto (consumare una mappa già costruita, playlist, lavagna, player,
; Folder analysis, i cue) gira, e le pagine coinvolte lo dicono da sole.
;
; Non firmato: per una distribuzione vera serve un certificato e
; `SignTool`, che non fa parte di questo script.

#define AppName "DjCaddy"
#define AppVersion "0.1.0"
#define AppPublisher "Maurizio Spadaccino"
#define AppExe "DjCaddy.exe"

[Setup]
AppId={{7B1C2E64-6D2A-4E1B-9E3F-0A2D5C8F41A7}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=..\dist
OutputBaseFilename={#AppName}-{#AppVersion}-setup
SetupIconFile=djcaddy.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Il bundle è grande: senza admin si installa sotto il profilo dell'utente,
; che è anche dove serve, perché DjCaddy non ha servizi né driver.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\{#AppName}\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; \
    Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Icona sul desktop"; \
    GroupDescription: "Collegamenti:"

[Run]
Filename: "{app}\{#AppExe}"; Description: "Avvia {#AppName}"; \
    Flags: nowait postinstall skipifsilent
