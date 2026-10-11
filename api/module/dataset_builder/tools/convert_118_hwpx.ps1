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
  [switch]$SkipRegister,
  [switch]$Unregister
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
if ($Unregister) {
  # HKCU 항목은 HKCR 병합에서 HKLM 보다 우선한다. 정식 등록(/regserver)을 하려면
  # 먼저 이걸 지워야 손으로 쓴 불완전한 등록이 정식 등록을 가리지 않는다.
  foreach ($k in @("HKCU:\Software\Classes\CLSID\$CLSID",
                   "HKCU:\Software\Classes\HWPFrame.HwpObject",
                   "HKCU:\Software\Classes\HWPFrame.HwpObject.2")) {
    if (Test-Path $k) { Remove-Item -Recurse -Force $k; Write-Host "removed $k" }
  }
  Write-Host "Done. Now run, from an ADMIN PowerShell:"
  Write-Host "  & `"$Exe`" /regserver"
  exit 0
}

if (-not $SkipRegister) {
  $server = '"{0}" -Automation' -f $Exe
  # 순서가 중요하다. New-Item -Force 는 이미 있는 키를 다시 만들면서 하위 키를 날린다.
  # 해시테이블은 순서가 없어서 부모가 자식보다 늦게 처리되면 방금 쓴 LocalServer32 가
  # 사라진다 — 실제로 그렇게 되어 "Class not registered" 가 났다. 부모부터 순서대로 쓴다.
  $keys = [ordered]@{
    "HKCU:\Software\Classes\CLSID\$CLSID"                          = 'HwpObject Class'
    "HKCU:\Software\Classes\CLSID\$CLSID\LocalServer32"            = $server
    "HKCU:\Software\Classes\CLSID\$CLSID\ProgID"                   = 'HWPFrame.HwpObject.2'
    "HKCU:\Software\Classes\CLSID\$CLSID\VersionIndependentProgID" = 'HWPFrame.HwpObject'
    "HKCU:\Software\Classes\CLSID\$CLSID\Programmable"              = ''
    "HKCU:\Software\Classes\CLSID\$CLSID\TypeLib"                   = '{7D2B6F3C-1D95-4E0C-BF5A-5EE564186FBC}'
    "HKCU:\Software\Classes\HWPFrame.HwpObject"                     = 'HwpObject Class'
    "HKCU:\Software\Classes\HWPFrame.HwpObject\CLSID"               = $CLSID
    "HKCU:\Software\Classes\HWPFrame.HwpObject\CurVer"              = 'HWPFrame.HwpObject.2'
    "HKCU:\Software\Classes\HWPFrame.HwpObject.2"                   = 'HwpObject Class'
    "HKCU:\Software\Classes\HWPFrame.HwpObject.2\CLSID"             = $CLSID
  }
  foreach ($k in $keys.Keys) {
    if (-not (Test-Path $k)) { New-Item -Path $k -Force | Out-Null }
    Set-ItemProperty -Path $k -Name '(Default)' -Value $keys[$k]
  }

  # 썼다고 믿지 말고 되읽는다. 앞서 조용히 날아간 적이 있다.
  $missing = @()
  foreach ($k in $keys.Keys) {
    # 기본값은 쓸 때 '(Default)', 읽을 때 '(default)' 로 나온다. 대소문자를 가리지 않게 읽는다.
    $item = Get-ItemProperty -Path $k -ErrorAction SilentlyContinue
    $v = if ($item) { $item.'(default)' } else { $null }
    if ($null -eq $v -and $item) { $v = $item.'(Default)' }
    if ($null -eq $v -or $v -ne $keys[$k]) { $missing += $k }
  }
  if ($missing.Count) {
    Write-Host "Registry write did not stick:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
  }
  Write-Host "COM registered under HKCU, all $($keys.Count) keys verified."
  Write-Host "  LocalServer32 = $server"
}

# ---- 객체 생성 확인. 실패하면 118번 헛돌지 않고 여기서 멈춘다 ----------------
try {
  $hwp = New-Object -ComObject HWPFrame.HwpObject
} catch {
  Write-Host "Failed to create COM object: $($_.Exception.Message)" -ForegroundColor Red
  if ($_.Exception.Message -match '80080005') {
    Write-Host ""
    Write-Host "The class resolves but Hancom will not start as a COM server." -ForegroundColor Yellow
    Write-Host "Hand-written HKCU keys are only a subset of what Hancom registers"
    Write-Host "(Programmable / TypeLib / implemented Categories). Use the supported route:"
    Write-Host ""
    Write-Host "  1) powershell -File `"$PSCommandPath`" -Unregister"
    Write-Host "  2) from an ADMIN PowerShell:  & `"$Exe`" /regserver"
    Write-Host "  3) re-run this script with -SkipRegister"
  } else {
    Write-Host "Check: (Get-ItemProperty 'HKCU:\Software\Classes\CLSID\$CLSID\LocalServer32').'(default)'"
  }
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
