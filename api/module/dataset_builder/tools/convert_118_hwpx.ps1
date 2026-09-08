<#
  0바이트로 생성된 계약서 PDF 118건을 HWPX 원본에서 다시 렌더링한다.

  배경: dataset/생성계약서.zip 안에서 이미 PDF 118건이 0바이트다(추출 실패가 아니라
  상류의 hwpx→pdf 변환 실패). HWPX 원본 118건은 전부 멀쩡하고 내용도 서로 다르다.

  반드시 32비트 PowerShell 로 실행할 것:
    C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy Bypass -File convert_118_hwpx.ps1

  왜 32비트인가: Hwp.exe 가 32비트 PE 라 COM 클래스가 32비트 레지스트리 뷰에만 유효하다.
  32비트 프로세스에서 HKCU:\Software\Classes\CLSID 에 쓰면 WOW64 가 알아서
  Wow6432Node 쪽으로 넘겨주므로, 등록과 사용을 같은 32비트 세션에서 하면 문제가 없다.

  관리자 권한은 필요 없다 — HKCU 에만 쓴다. 되돌리려면:
    Remove-Item -Recurse HKCU:\Software\Classes\CLSID\{965829DB-438E-4d31-B4FA-F1F8819A35FD}
    Remove-Item -Recurse HKCU:\Software\Classes\HWPFrame.HwpObject
#>

param(
  [string]$In   = "C:\Users\user\AppData\Local\Temp\hwpx_convert\in",
  [string]$Out  = "C:\Users\user\AppData\Local\Temp\hwpx_convert\out",
  [string]$Exe  = "C:\Program Files (x86)\Hnc\Office 2024\HOffice130\Bin\Hwp.exe",
  [switch]$SkipRegister
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
$CLSID = '{965829DB-438E-4d31-B4FA-F1F8819A35FD}'

if ([Environment]::Is64BitProcess) {
  Write-Host "This script must run under 32-bit PowerShell." -ForegroundColor Red
  Write-Host "  C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy Bypass -File `"$PSCommandPath`""
  exit 1
}
if (-not (Test-Path $Exe)) { Write-Host "Hwp.exe not found: $Exe" -ForegroundColor Red; exit 1 }

# ---- COM 등록 (HKCU, 사용자 단위) -------------------------------------------
# reg.exe 로 하면 경로에 공백과 따옴표가 섞여 인용이 깨진다("ERROR: Invalid syntax").
# PowerShell 네이티브 cmdlet 은 인용을 직접 다루지 않으므로 그 문제가 없다.
if (-not $SkipRegister) {
  $server = '"{0}" -Automation' -f $Exe
  $keys = @{
    "HKCU:\Software\Classes\CLSID\$CLSID"                          = 'HwpObject Class'
    "HKCU:\Software\Classes\CLSID\$CLSID\LocalServer32"            = $server
    "HKCU:\Software\Classes\CLSID\$CLSID\ProgID"                   = 'HWPFrame.HwpObject.2'
    "HKCU:\Software\Classes\CLSID\$CLSID\VersionIndependentProgID" = 'HWPFrame.HwpObject'
    "HKCU:\Software\Classes\HWPFrame.HwpObject"                    = 'HwpObject Class'
    "HKCU:\Software\Classes\HWPFrame.HwpObject\CLSID"              = $CLSID
    "HKCU:\Software\Classes\HWPFrame.HwpObject\CurVer"             = 'HWPFrame.HwpObject.2'
    "HKCU:\Software\Classes\HWPFrame.HwpObject.2"                  = 'HwpObject Class'
    "HKCU:\Software\Classes\HWPFrame.HwpObject.2\CLSID"            = $CLSID
  }
  foreach ($k in $keys.Keys) {
    New-Item -Path $k -Force | Out-Null
    Set-ItemProperty -Path $k -Name '(Default)' -Value $keys[$k]
  }
  Write-Host "COM registered under HKCU. LocalServer32 = $server"
}

# ---- 객체 생성 확인. 실패하면 118번 헛돌지 않고 여기서 멈춘다 ----------------
try {
  $hwp = New-Object -ComObject HWPFrame.HwpObject
} catch {
  Write-Host "Failed to create COM object: $($_.Exception.Message)" -ForegroundColor Red
  Write-Host "Check: (Get-ItemProperty 'HKCU:\Software\Classes\CLSID\$CLSID\LocalServer32').'(default)'"
  exit 1
}
Write-Host "HWPFrame.HwpObject created OK"

# 보안 모듈이 없으면 파일을 열 때마다 모달 대화상자가 뜬다.
try { $hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModuleExample") | Out-Null } catch {}
try { $hwp.XHwpWindows.Item(0).Visible = $false } catch {}

if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out -Force | Out-Null }

$ok = 0; $fail = 0; $bad = @()
foreach ($f in Get-ChildItem "$In\*.hwpx") {
  $dst = Join-Path $Out ($f.BaseName + ".pdf")
  try {
    $null = $hwp.Open($f.FullName, "HWPX", "forceopen:true")
    $null = $hwp.SaveAs($dst, "PDF", "")
    $hwp.Clear(1)
    if ((Test-Path $dst) -and (Get-Item $dst).Length -gt 0) { $ok++ }
    else { $fail++; $bad += $f.BaseName }
  } catch {
    $fail++; $bad += $f.BaseName
    Write-Host ("FAIL {0} - {1}" -f $f.Name, $_.Exception.Message)
  }
}
try { $hwp.Quit() } catch {}

Write-Host ""
Write-Host ("converted={0} failed={1}  ->  {2}" -f $ok, $fail, $Out)
if ($bad.Count) { Write-Host ("failed ids: {0}" -f ($bad -join ', ')) }
Write-Host "Every good contract PDF is exactly 5 pages. Reject any output that is not."
