# AC61 protocol correction

`imaging_protocol_rest.csv` is shifted one calendar day forward relative to the
date encoded in the raw folder name. EXIF timestamps in `DSC22047.JPG` and
`DSC28361.JPG` show that the resting sequence runs from 2023-07-11 to
2023-08-02, not 2023-07-10 to 2023-08-01.

Without this correction, the resting sequence appears roughly 24 hours before
the injection sequence and calibration frames resolve against the wrong
experiment time.
