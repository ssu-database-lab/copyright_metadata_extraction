# Windows TDR (Timeout Detection & Recovery) 타임아웃 증가 스크립트
# 관리자 권한으로 실행해야 합니다.
#
# 사용법:
#   1. PowerShell을 관리자 권한으로 실행
#   2. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   3. .\fix_tdr_timeout.ps1
#
# 적용 후 컴퓨터를 재시작해야 합니다.

$regPath = "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers"

# TdrDelay: GPU 커널 최대 실행 허용 시간 (기본 2초 → 60초)
Set-ItemProperty -Path $regPath -Name "TdrDelay" -Value 60 -Type DWord -Force
Write-Host "[OK] TdrDelay = 60초 설정 완료"

# TdrDdiDelay: 드라이버 응답 대기 시간 (기본 5초 → 60초)
Set-ItemProperty -Path $regPath -Name "TdrDdiDelay" -Value 60 -Type DWord -Force
Write-Host "[OK] TdrDdiDelay = 60초 설정 완료"

Write-Host ""
Write-Host "설정이 완료되었습니다. 컴퓨터를 재시작하세요."
Write-Host "재시작 후 적용됩니다."
