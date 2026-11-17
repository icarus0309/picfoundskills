param(
  [string]$Root='.', [string]$Out='', [int]$Limit=10, [int]$Offset=0,
  [int]$Workers=8, [int]$MaxCalls=2000,
  [ValidateSet('run','prepare','screen','verify','export','evaluate')][string]$Stage='run'
)
$ErrorActionPreference='Stop'
if(-not $Out){$Out=Join-Path $Root ('image-matching/runs/batch-{0:D4}-{1:D4}' -f $Offset,$Limit)}
$scriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
& python -X utf8 (Join-Path $scriptRoot 'run.py') --root $Root --out $Out --offset $Offset --limit $Limit --workers $Workers --max-calls $MaxCalls --stage $Stage
if($LASTEXITCODE -ne 0){throw "Stage $Stage failed; rerun with the same arguments to resume"}
