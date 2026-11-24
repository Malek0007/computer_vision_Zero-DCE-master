# Training and Evaluation Report

## 6. Training Execution and Verification

### Training configuration

The repository's `lowlight_train.py` defaults to Adam with learning rate 0.0001, weight decay 0.0001, batch size 8, four data-loader workers, and 200 epochs. The loader discovers `2,002` JPEG training images and resizes each to 256 × 256 pixels. With batch size 8, this dataset yields 251 iterations per epoch. The `snapshots_100epochs` folder contains checkpoints named `Epoch0.pth` through `Epoch99.pth`. Other checkpoint directories contain `snapshots_training_test/Epoch0.pth`, `snapshots_10epochs/Epoch0.pth` through `Epoch9.pth`, and `snapshots/Epoch0.pth` through `Epoch3.pth`; `snapshots/Epoch99.pth` is the authors' pretrained checkpoint used as the baseline. These are separate runs and are not combined into one epoch count. The original command line and run configuration were not saved, so defaults describe the script and may not prove every runtime argument.

### Hardware and environment

The training code selects Apple MPS when PyTorch reports it available and otherwise selects CPU. No terminal log or run metadata records which device was active during training. The artifact verification and inference in this report ran on **cpu** (Python 3.14.3, PyTorch 2.15.0; MPS available: False). This current runtime does not establish the historical training device.

### Completed epochs and iterations

The script saves checkpoints every 10 batches by default, using the current zero-based epoch index as the filename. Assuming these defaults were used, the checkpoint series proves that training progressed through 99 complete epochs: each checkpoint before the final index is followed by a checkpoint for the next epoch. `Epoch99.pth` proves the final recorded epoch reached batch 250 of 251. The checkpoint cadence alone cannot establish whether its final batch completed. Thus the conservative evidence is **99 complete epochs, with the next epoch reaching at least 250/251 iterations** (at least 25,099 optimizer steps under the script's default batch size). The original CLI arguments and terminal log are unavailable, so this count is conditional on the script defaults. No training log is present to resolve the final iteration or intended epoch count.

### Checkpoint verification

The latest user-training checkpoint is `snapshots_100epochs/Epoch99.pth`; it loaded strictly into the repository's unchanged `enhance_net_nopool` architecture, containing 79,416 parameters. The authors' `snapshots/Epoch99.pth` checkpoint also loaded strictly. Both models processed the three RGB test images and returned finite outputs with their input dimensions. SHA-256 hashes and file sizes are recorded in `checkpoint_verification.txt`. The authors' checkpoint was read only and its hash remained unchanged during artifact generation.

### Training-loss analysis

No historical training-loss log or per-epoch loss file was found. The existing CSV in `comparison_results` contains image evaluation values rather than training losses. Therefore, initial/final losses and a training-loss curve cannot be reported; no `training_loss.png` was generated.

## 7. Evaluation of the Trained Model

### Test set and inference procedure

Evaluation used DICM images 26, 29, and 63 from `data/test_data/DICM`. The same native-resolution RGB image was converted to float32 in [0, 1] and passed as a batch of one to both checkpoints. Outputs were clipped to [0, 1] before saving and measuring. Processing time includes the model forward pass, excludes one warm-up pass, and is the mean of 3 timed repetitions per image and model. The table reports luminance mean and population standard deviation using BT.709 coefficients, on a 0–255 scale. No matched normal-light references were found for these DICM images, so PSNR and SSIM were not calculated.

### Original, pretrained, and trained comparisons

The comparison figures show the original input, the authors' pretrained checkpoint, and the user's `snapshots_100epochs/Epoch99.pth` checkpoint side by side. These are fresh outputs from the checkpoints named above, not prior saved result images.

### Quantitative measurements

| Image | Method | Mean brightness | Luminance contrast (SD) | Mean inference time (s) |
|---|---|---:|---:|---:|
| DICM 26 | Original | 23.15 | 44.87 | — |
| DICM 26 | Authors' Pretrained | 40.08 | 64.34 | 0.255273 |
| DICM 26 | My Trained Model | 44.00 | 67.32 | 0.255690 |
| DICM 29 | Original | 31.43 | 49.43 | — |
| DICM 29 | Authors' Pretrained | 85.32 | 55.15 | 0.465191 |
| DICM 29 | My Trained Model | 96.39 | 54.06 | 0.481424 |
| DICM 63 | Original | 159.12 | 30.42 | — |
| DICM 63 | Authors' Pretrained | 185.59 | 22.73 | 0.296677 |
| DICM 63 | My Trained Model | 189.50 | 22.14 | 0.295733 |

The measured average forward time was 0.339047 seconds per image for the authors' checkpoint and 0.344282 seconds per image for the user-trained checkpoint on cpu. Brightness and contrast are descriptive image statistics; higher values do not by themselves establish better perceptual quality. Processing time depends on the local hardware and software environment.

### Observed differences

Across these three images, the user-trained checkpoint has higher mean luminance than the authors' checkpoint by 3.92, 11.07, and 3.91 points for DICM 26, 29, and 63, respectively. Its luminance standard deviation is higher on DICM 26 and slightly lower on DICM 29 and 63. Visually, the two model outputs are similar for DICM 26 and 63; DICM 29 becomes substantially brighter, while visible grain is also lifted in the darker regions. The outputs therefore show a stronger brightness adjustment on some inputs, with noise amplification as a limitation. These measurements do not quantify color fidelity, detail recovery, or subjective naturalness, and the three-image sample is not a benchmark-level quality evaluation.

## 8. Limitations and Conclusion

The paper reports training on 2,422 multi-exposure SICE Part 1 images resized to 512 × 512, using an NVIDIA 2080 Ti, batch size 8, Adam, and a fixed learning rate of 1 × 10⁻⁴. The local project contains 2,002 training JPEGs and its loader resizes them to 256 × 256. The script defaults also set weight decay to 1 × 10⁻⁴, whereas the paper describes Adam with its default parameters. The surviving checkpoint sequence supports 99 full epochs and nearly all of another epoch, but does not prove the last iteration, the original command-line target, or the historical training device. The current code's default epoch target is 200, which further shows why the folder name alone cannot establish the configured target. [Guo et al., CVPR 2020](https://staff.ie.cuhk.edu.hk/~ccloy/files/cvpr_2020_zerodce.pdf).

Low-light enhancement can amplify sensor noise and may over-enhance bright regions. Brightness and luminance contrast summarize exposure changes but cannot determine whether the result looks natural or preserves fine detail. No correctly matched reference image was available for the selected DICM inputs, so full-reference metrics were omitted. The experiment demonstrates that the saved user-trained weights are compatible with the Zero-DCE architecture and produce measurable enhancements on three DICM images; it does not establish parity with the paper's data, training setup, hardware, or benchmark results.
