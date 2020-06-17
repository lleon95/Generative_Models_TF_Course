# Day 2: Improve performances

## Topics:

- TF execution modes
- GPUs support
- Improve performances with input data pipelines
- Parallel execution across multiple devices
- Distribution strategies

## Instructions to access the cluster

To launch the interactive apps via the Open OnDemand portal follow the link:

[```https://login-web.hpc.cam.ac.uk/```](https://login-web.hpc.cam.ac.uk/)

If needed, you can access a login node via ssh:

```
ssh <username>@login-gpu.hpc.cam.ac.uk
```

### Obtain the credentials

1. **Make sure to have correctly installed Google Authenticator** (or any other OTP 
authentication app).

2. I will give each one of you an unique link via private message. To reveal the password
 use the passphrase given during the course. **The password is revealed only once!** Store
 it somewhere safe.

3. Login to the [OOD portal](https://login-web.hpc.cam.ac.uk/) using username / password.

4. Scan the QR code using Google Authenticator to pair your device as second factor
authenticator.

5. Insert the one time password provided by the authentication app.

### Launch an Interactive Session

From the Dashboard select the "Interactive Apps" tab, then "Jupyter Notebook" or 
"TensorBoard". Fill the fields as follows:

| Fields | Jupyter Notebook | TensorBoard |
|-----|-----|-----|
|Project Account| training-gpu | training-cpu |
| Partition | pascal | skylake |
| Reservation * | dltraining_day1 | |
| Number of hours ** | <=12 | |
| Number of ... | <=4 | 1 |

To launch TensorBoard on a specific folder fill the Logdir field accordingly. 

\* `dltraining_day1` for Tuesday, `dltraining_day2` for Wednesday, `dltraining_day3` for Thursday.

\** Remember that the reservation is from 10am - 10pm CEST
