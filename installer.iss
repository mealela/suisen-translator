; 水仙翻译器 Suisen Translator — Inno Setup 安装包脚本
; 编译: ISCC.exe installer.iss  (产出在 水仙\ 目录)
; 用户态安装: 不需要管理员权限, 默认装到 %LOCALAPPDATA%\Programs\suisen
; (程序会在自身目录写 config/models/logs, 不能装 Program Files)

#define MyAppName "水仙翻译器 Suisen Translator"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "mealela"
#define MyAppExe "suisen.exe"

[Setup]
AppId={{7E3A9C51-2F4B-4D8A-B6E1-9A0C5D2F8E44}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}
VersionInfoDescription={#MyAppName} 安装程序
DefaultDirName={localappdata}\Programs\suisen
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=水仙
OutputBaseFilename=suisen-setup-{#MyAppVersion}
SetupIconFile=assets\logo.ico
UninstallDisplayIcon={app}\{#MyAppExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
DisableWelcomePage=no
DisableDirPage=no
InfoBeforeFile=installer-notice.txt
MinVersion=10.0

[Languages]
Name: "chs"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "在桌面创建快捷方式"; GroupDescription: "附加选项:"

[Files]
; 只打程序本体: 排除 models(模型库首用再下载, ~16GB)/logs/.work/config.json(含密钥)
Source: "dist\suisen\suisen.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\suisen\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
; 视频播放组件 libmpv 一并打入, 新机器免下载
Source: "dist\suisen\models\bin\libmpv-2.dll"; DestDir: "{app}\models\bin"; Flags: ignoreversion
; 许可证与第三方声明(LGPL/GPL 组件合规要求随包附许可证文本)
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "THIRD-PARTY-NOTICES.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "third_party_licenses\*"; DestDir: "{app}\third_party_licenses"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; IconFilename: "{app}\{#MyAppExe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; IconFilename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
