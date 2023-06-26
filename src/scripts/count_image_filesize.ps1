# Using PS to copy the .jpg files

$source = 'C:\Users\ad_xleong\Desktop\coral-sleuth\images\MCR_LTER_ComputerVision_LabeledCorals_2008_2009_2010\2010'
$destination = 'C:\Users\ad_xleong\Desktop\coral-sleuth\images\'
Get-ChildItem -Path $source -Filter *.jpg | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $destination
}

