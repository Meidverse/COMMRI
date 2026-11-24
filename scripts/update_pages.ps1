# Opens the report in Word, fills List of Figures / List of Tables page numbers from the real layout,
# refreshes the Contents page, saves, and exports a PDF preview.
param([string]$Docx, [string]$Pdf)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Docx)) { throw "File not found: $Docx" }
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$doc = $w.Documents.Open($Docx)
function Get-PageOf($text) {
  $r = $doc.Content
  $f = $r.Find
  $f.ClearFormatting()
  if ($f.Execute($text)) { return $r.Information(1) } else { return $null }  # 1 = wdActiveEndAdjustedPageNumber
}
for ($pass = 1; $pass -le 2; $pass++) {
  $doc.Repaginate()
  # List of Figures: table 1, rows 2..n
  $lof = $doc.Tables.Item(1)
  for ($i = 2; $i -le $lof.Rows.Count; $i++) {
    $no = ($lof.Cell($i, 1).Range.Text -replace '[\r\a]', '').Trim() -replace 'Figure ', 'Fig. '
    $pg = Get-PageOf("${no}:")
    if ($pg) { $lof.Cell($i, 3).Range.Text = "$pg" } else { Write-Output "NOT FOUND $no" }
  }
  # List of Tables: table 2 rows map to body tables 4..10 (after the notation table)
  $lot = $doc.Tables.Item(2)
  for ($i = 2; $i -le $lot.Rows.Count; $i++) {
    $t = $doc.Tables.Item($i + 2)
    $lot.Cell($i, 3).Range.Text = "$($t.Range.Information(1))"
  }
  if ($doc.TablesOfContents.Count -gt 0) { $doc.TablesOfContents.Item(1).Update() }
}
for ($i = 2; $i -le $lof.Rows.Count; $i++) { Write-Output (($lof.Rows.Item($i).Range.Text -replace '[\r\a]+', ' | ').Trim()) }
for ($i = 2; $i -le $lot.Rows.Count; $i++) { Write-Output (($lot.Rows.Item($i).Range.Text -replace '[\r\a]+', ' | ').Trim()) }
$doc.Save()
$doc.SaveAs2([ref]$Pdf, [ref]17)
Write-Output "pages: $($doc.ComputeStatistics(2))"
$doc.Close($false)
$w.Quit()
