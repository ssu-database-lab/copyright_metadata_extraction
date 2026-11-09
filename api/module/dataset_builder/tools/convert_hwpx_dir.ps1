<#
  폴더 안의 *.hwpx 를 한글(HWPFrame.HwpObject)로 열어 같은 이름의 PDF 로 저장한다.
  계약서 생성기 v5 (generate_contracts_v5.py) 가 묶음마다 부른다.

  32비트 PowerShell 로 실행 (Hwp.exe 가 32비트라 COM 이 32비트 레지스트리 뷰에만 있다):
    C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass `
      -File convert_hwpx_dir.ps1 -In C:\...\in -Out C:\...\out
  COM 등록은 convert_118_hwpx.ps1 에서 이미 했다 (HKCU). 여기서는 하지 않는다.
  이미 PDF 가 있으면 건너뛴다 (-Force 로 다시 만든다). 끝에 "converted=N failed=M skipped=K" 를 찍는다.
#>
param(
  [Parameter(Mandatory = $true)][string]$In,
  [Parameter(Mandatory = $true)][string]$Out,
  [switch]$Force
)
$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
if ([Environment]::Is64BitProcess) { Write-Host "Run under 32-bit PowerShell (SysWOW64)." -ForegroundColor Red; exit 1 }

$hwp = New-Object -ComObject HWPFrame.HwpObject
try { $hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModuleExample") | Out-Null } catch {}
try { $hwp.XHwpWindows.Item(0).Visible = $false } catch {}
if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out -Force | Out-Null }

$ok = 0; $fail = 0; $skip = 0; $i = 0
$files = @(Get-ChildItem -LiteralPath $In -Filter *.hwpx)
foreach ($f in $files) {
  $i++
  $dst = Join-Path $Out ($f.BaseName + ".pdf")
  if ((-not $Force) -and (Test-Path -LiteralPath $dst) -and (Get-Item -LiteralPath $dst).Length -gt 0) { $skip++; continue }
  try {
    $null = $hwp.Open($f.FullName, "HWPX", "forceopen:true")
    $null = $hwp.SaveAs($dst, "PDF", "")
    $hwp.Clear(1)
    if ((Test-Path -LiteralPath $dst) -and (Get-Item -LiteralPath $dst).Length -gt 0) { $ok++ }
    else { $fail++; Write-Host ("FAIL {0} - empty output" -f $f.Name) }
  } catch {
    $fail++
    Write-Host ("FAIL {0} - {1}" -f $f.Name, $_.Exception.Message)
    try { $hwp.Clear(1) } catch {}
  }
  if ($i % 50 -eq 0) { Write-Host ("progress {0}/{1}" -f $i, $files.Count) }
}
try { $hwp.Quit() } catch {}
Write-Host ("converted={0} failed={1} skipped={2}" -f $ok, $fail, $skip)
