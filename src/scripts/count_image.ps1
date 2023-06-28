$source = 'C:\Users\ad_xleong\Desktop\coral-sleuth\data\images'

(Get-ChildItem -Path $source -Include *.png,*.jpg -Recurse).Count
