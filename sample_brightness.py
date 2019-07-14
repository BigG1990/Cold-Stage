import json
import sys
import string
from PIL import Image, ImageDraw
from progress.bar import IncrementalBar
import os
from matplotlib import pyplot
import csv

## Calibration Tool: https://codepen.io/backoefer/full/axbmyG

# Settings
file_path = '/Users/damon/Documents/GitHub/FreezingDevices/WW20/' # Path to image files
log_file = file_path + '20190418_T_log_WW_20.csv'
brightness_probe_size = 29 # Needs to be uneven, so the circle's mid point is defined by a single pixel and corners are all equally large

# Calibration
delta_unfrozen_treshold = 0.2 # E.g. 0.2 = If total brightness difference of a probe is lower than 0.2 of the largest brightness difference, consider it unfrozen. This is particularly necessary when working with partial datasets
delta_significance_threshold = 0.5 # The relative strength of a delta brightness change required to count as frozen. For example, with this setting set to 0.2, we will recognize the first delta in the top 80% of deltas as the freezing one

### FUNCTION DEFINITIONS ###

# Function to get calibration file
def get_calibration():
  
  # Script looks for a 'default_calibration.json' in the folder it is run in, but uses a 'calibration.json' in the test folder if available
  calibration = './default_calibration.json'
  custom_file = file_path + 'calibration.json'
  
  if os.path.isfile(custom_file):
    print('Using test-specific calibration...')
    calibration = custom_file
  elif os.path.isfile(calibration):
    print('Using default calibration file...')
  else:
    print('ERROR: No calibration file found, aborting')
    sys.exit()
  
  with open(calibration, 'r') as f:
    calibration = json.load(f)
    coordinates = calibration['probe_coordinates']
    return coordinates

# Function to get all images in the image folder
def get_images():
    
    print('Open image files...')

    folder_contents = os.listdir(file_path)
    image_extensions = ('.jpg', '.jpeg')
    image_files = []
    for i, filename in enumerate(folder_contents):
      file_is_visible = filename[:1] != '.'
      file_exists = os.path.isfile(file_path + filename)
      file_is_image = any(ext in filename for ext in image_extensions)
      if file_exists and file_is_image and file_is_visible:
        image_files.append(filename)
    return sorted(image_files) # Needs to be sorted, as files aren't read in ascending order by default


# Create a circular mask that is used to constrain the sampled image area to a circle
def make_probe_mask(size):
  mask_image = Image.new('L', (size, size), 'black') # Make black image
  drawing_context = ImageDraw.Draw(mask_image)
  drawing_context.ellipse([0, 0, size, size], 'white') # Draw white circle on it
  return mask_image

brightness_probe_mask = make_probe_mask(brightness_probe_size)


# Use the histogram of a grayscale image to calculate the mean brightness between 0.0 (black) and 1.0 (white)
def calc_mean_brightness(histogram):
  pixel_count = float(sum(histogram))
  val = 0.0
  for i, n in enumerate(histogram):
    val = val + (float(n) * float(i)) / pixel_count
  normalized_val = val / 255.0 # Convert from 0-255 to 0.0-1.0
  return normalized_val


# Take a circular sample of pixels at the given coordinates
def probe_image_at_point(image, x, y):
  
  # Determine probe bounds
  half_size = float(brightness_probe_size) / 2.0
  probe_bounds = (x-half_size, y-half_size, x+half_size, y+half_size)

  # Crop probe sample
  cropped_image = image.crop(probe_bounds)

  # Create histogram with mask
  histogram = cropped_image.histogram(brightness_probe_mask)

  # Calculate mean gray
  mean_brightness = calc_mean_brightness(histogram)
  return mean_brightness


# Utility function for getting individual columns from two-dimensional arrays
def get_column(matrix, i):
    return [row[i] for row in matrix]


# Get temperatures from log file
def get_temperatures():
  with open(log_file, 'rU') as infile:
    csv_contents = csv.reader(infile, delimiter=';')
    temps = get_column(csv_contents, 2)
    # Convert to floats
    for i, temp in enumerate(temps):
      temps[i] = float(temp)
    return temps


# Save an overlay of the sample grid to verify sample locations
def save_sample_map(images):
  last_image = Image.open(file_path + images[len(images)-1]) # Get last image as image object
  drawing_context = ImageDraw.Draw(last_image)

  # Half size
  half_size = float(brightness_probe_size) / 2.0
  
  # Draw red dots in sample locations
  for i, coords in enumerate(probe_coordinates):
    x = coords['x']
    y = coords['y']
    ellipse_bounds = [x-half_size, y-half_size, x+half_size, y+half_size]
    color = 'cyan'
    if probe_validity[i] == False:
      color = 'red'
    drawing_context.ellipse(ellipse_bounds, color) # Draw red circle

  # Save test image
  last_image.save(file_path + 'sample_map.png', 'PNG')


# Take brightness samples for whole image set
def measure_brightness(images):
  rows = []
  bar = IncrementalBar('Measure brightness...', max=len(images), suffix='%(percent).1f%% - %(eta)ds')
  for i, image in enumerate(images):
    im = Image.open(file_path + image).convert(mode='L');
    cols = []
    for j, coords in enumerate(probe_coordinates):
      x = coords['x']
      y = coords['y']
      brightness = probe_image_at_point(im, x, y)
      cols.append(brightness)
    rows.append(cols)
    bar.next()

  bar.finish()
  return rows


# Reject unfrozen probes
def reject_unfrozen_probes(matrix):

  # Find total brightness delta for each probe
  brightness_deltas = [0.0 for x in range(96)]
  for i, probe in enumerate(matrix[0]):
    probes = get_column(matrix, i)
    min_brightness = min(probes)
    max_brightness = max(probes)
    brightness_deltas[i] = max_brightness - min_brightness
  
  # Get smalles/largest delta across all probes
  max_delta = max(brightness_deltas)

  # Reject probes that don't have sufficient brightness delta
  rejected_count = 0
  probe_validity = [True for x in range(96)]
  invalid_probes = []
  for i, delta in enumerate(brightness_deltas):
    delta_percentile = delta/max_delta
    if delta_percentile <= delta_unfrozen_treshold:
      invalid_probes.append(probe_coordinates[i]['id'])
      rejected_count = rejected_count + 1
      probe_validity[i] = False

  if (rejected_count > 0):
    # TODO: Mention which ones don't freeze
    print('WARNING: Rejected ' + str(rejected_count) + ' sample as unfrozen: ' + ' , '.join(invalid_probes)) # Show which probes are invalid
  
  return probe_validity


# Determine frozen sample count for all images
def analyze_data_incremental(matrix, images):

  print('Analyzing data...')

  mins = [None for x in range(96)]
  maxs = [0.0 for x in range(96)]

  # Empty 2 dimensional matrix to store delta values for all probes in all images
  deltas = [[0.0 for probe in range(96)] for image in range(len(images))]

  # Determine delta changes for all measured brightnesses
  for r, row in enumerate(matrix):
    for c, col in enumerate(row):
      val = 0
      if r != 0:

        # Process delta brightness changes between images
        brightness_delta = matrix[r][c] - matrix[r-1][c] # Brightness delta for this probe, image over image
        brightness_delta = min(brightness_delta, 0) # Ignore positive delta changes
        brightness_delta = abs(brightness_delta) # Make delta asolute
        val = brightness_delta
        
        # Determine min and max deltas for this probe
        if val > maxs[c]:
          maxs[c] = val
        if mins[c] == None or val < mins[c]:
          mins[c] = val

      deltas[r][c] = val

  
  # Determine frozen counts
  frozen_counts = []
  frozen_memory = [False for x in range(96)]
  newly_frozen_ids = ['' for x in range(len(matrix))]

  for i, row in enumerate(deltas):
    frozen_count = 0
    for j, sample in enumerate(row):

      # Skip probes that never freeze
      if probe_validity[j] == False:
       continue

      # Determine percentile of this delta change
      sample_percentile = (sample-mins[j])/(maxs[j]-mins[j])

      newly_frozen = sample_percentile >= delta_significance_threshold and not frozen_memory[j]

      if newly_frozen:
        if newly_frozen_ids[i] is not '':
          newly_frozen_ids[i] += ', '
        newly_frozen_ids[i] += probe_coordinates[j]['id']
      
      # If the delta change is higher than the defined threshold percentile, mark as frozen
      if newly_frozen or frozen_memory[j]:
        frozen_count = frozen_count + 1
        if not frozen_memory[j]:
          frozen_memory[j] = True

    frozen_counts.append(frozen_count)

  return (frozen_counts, newly_frozen_ids)


# Plot results
def plot_results(temperatures, frozen_counts):

  print('Graphing results and saving as graph.png...')

  pyplot.xlim(0.0, round(min(temperatures)) - 2.0)
  pyplot.ylabel('fraction of frozen samples')
  pyplot.xlabel('Temperatur in C')
  pyplot.plot(temperatures, [float(x)/96 for x in frozen_counts], '#E24A33')
  pyplot.savefig(file_path + 'graph.png', dpi=150, bbox_inches='tight')


# Save Results to CSV file
def save_results(images, temperatures, frozen_counts, newly_frozen_ids):
  
  print('Saving output.csv...')
  
  csv_headers = ['Image Filename', 'Temperature', 'Frozen Count', 'Frozen Fraction', 'Newly Frozen #', 'Newly Frozen IDs']
  csv_rows = [csv_headers]

  prev_frozen_count = 0
  for i, image_name in enumerate(images):
    newly_frozen = frozen_counts[i] - prev_frozen_count
    csv_rows.append([image_name, temperatures[i], frozen_counts[i], frozen_counts[i]/96.0, newly_frozen, newly_frozen_ids[i]])
    prev_frozen_count = frozen_counts[i]
  
  with open(file_path + 'output.csv', 'w') as outfile:
    writer = csv.writer(outfile, lineterminator='\n')
    writer.writerows(csv_rows)
    outfile.close()


# ### SCRIPT ###

# # 0. Get Calibration
probe_coordinates = get_calibration()

# # 1. Get image files + log file with temperatures
all_images = get_images()
temperatures = get_temperatures()

# # 2. Measure brightness for all samples across all images
brightness_data = measure_brightness(all_images)
probe_validity = reject_unfrozen_probes(brightness_data) # Check for permanently unfrozen probes
save_sample_map(all_images)

# # 3. Determine when each sample froze
frozen_sample_counts, newly_frozen_ids = analyze_data_incremental(brightness_data, all_images)

# # 4. Plot results
plot_results(temperatures, frozen_sample_counts)

# # 5. Saving results
save_results(all_images, temperatures, frozen_sample_counts, newly_frozen_ids)

