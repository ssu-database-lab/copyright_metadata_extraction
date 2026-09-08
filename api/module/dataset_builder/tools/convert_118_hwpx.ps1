# 0바이트로 생성된 계약서 PDF 118건을 HWPX 원본에서 다시 렌더링한다.
#
# 배경: dataset/생성계약서.zip 안에서 이미 PDF 118건이 0바이트다(추출 실패가 아니라
# 상류의 hwpx→pdf 변환 실패). HWPX 원본은 118건 모두 멀쩡하며 각자 다른 내용을 담고 있다.
#
# 반드시 32비트 PowerShell 로 실행할 것:
#   C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -File convert_118_hwpx.ps1
# Hwp.exe 가 32비트라 COM 클래스가 32비트 레지스트리 뷰에만 등록되기 때문이다.
#
# 사전 준비 (둘 중 하나, 한 번만):
#   A. 관리자 권한 없이 — 현재 사용자만 등록 (권장, reg delete 로 되돌릴 수 있음)
#      $C='{965829DB-438E-4d31-B4FA-F1F8819A35FD}'
#      $E='C:\Program Files (x86)\Hnc\Office 2024\HOffice130\Bin\Hwp.exe'
#      reg add "HKCU\Software\Classes\HWPFrame.HwpObject" /ve /d "Hwp Object" /f
#      reg add "HKCU\Software\Classes\HWPFrame.HwpObject\CLSID" /ve /d $C /f
#      reg add "HKCU\Software\Classes\CLSID\$C" /ve /d "Hwp Object" /f
#      reg add "HKCU\Software\Classes\CLSID\$C\LocalServer32" /ve /d "`"$E`" -Automation" /f
#      reg add "HKCU\Software\Classes\CLSID\$C\ProgID" /ve /d "HWPFrame.HwpObject" /f
#   B. 관리자 PowerShell 에서
#      & "C:\Program Files (x86)\Hnc\Office 2024\HOffice130\Bin\Hwp.exe" /regserver
#
# 이미 등록된 HwpAutomationApp2.HwpAutomation 은 쓰지 말 것. 인스턴스는 만들어지지만
# 32비트 in-proc 서버라 Open() 에서 AccessViolationException 으로 죽는다(실측).

param(
  [string]$In  = "C:\Users\user\AppData\Local\Temp\hwpx_convert\in",
  [string]$Out = "C:\Users\user\AppData\Local\Temp\hwpx_convert\out"
)

if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out -Force | Out-Null }

$hwp = New-Object -ComObject HWPFrame.HwpObject
# 보안 모듈이 없으면 파일을 열 때마다 모달 대화상자가 뜬다. 한컴 개발자 사이트의
# FilePathCheckerModuleExample.dll 을 등록해 두면 조용히 지나간다.
try { $hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModuleExample") | Out-Null } catch {}
try { $hwp.XHwpWindows.Item(0).Visible = $false } catch {}

$ok = 0; $fail = 0
foreach ($f in Get-ChildItem "$In\*.hwpx") {
  $dst = Join-Path $Out ($f.BaseName + ".pdf")
  try {
    $null = $hwp.Open($f.FullName, "HWPX", "forceopen:true")
    $null = $hwp.SaveAs($dst, "PDF", "")
    $hwp.Clear(1)
    if ((Test-Path $dst) -and (Get-Item $dst).Length -gt 0) { $ok++ } else { $fail++; Write-Host "EMPTY: $($f.Name)" }
  } catch {
    $fail++; Write-Host "FAIL : $($f.Name) — $($_.Exception.Message)"
  }
}
try { $hwp.Quit() } catch {}
Write-Host "converted=$ok failed=$fail  ->  $Out"
Write-Host "다음: 5쪽이 아닌 산출물은 렌더링 불일치이므로 받아들이지 말 것 (정상 PDF 는 전부 5쪽)."
