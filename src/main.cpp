// PlatformIO main.cpp — Sign Detection Model (Optimized)
// Place this file in: src/main.cpp

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <cmath>

// ─── Configuration ───────────────────────────────────────────────────────────
#define ENERGY_START_THRESHOLD  0.6f    // gyro magnitude to START accumulating frames (was 0.3)
#define ENERGY_END_THRESHOLD    0.25f   // gyro magnitude to END a segment (was 0.15)
#define DEBOUNCE_MS             250     // ms of quiet before segment ends (was 300)
#define MIN_SEGMENT_MS          100     // ignore segments shorter than this (was 150)
#define SAMPLE_INTERVAL_MS      10      // target ~100Hz
#define FRAMES_TO_START         8       // consecutive active frames before segment starts (was 10)
#define STABILITY_THRESHOLD     1       // stability classifier value for STABLE (0=unknown, 1=stable, 2=transitioning)

// ─── Globals ─────────────────────────────────────────────────────────────────
Adafruit_BNO08x bno;
sh2_SensorValue_t sensorValue;

struct Frame {
  float real, i, j, k;
  float accel_x,  accel_y,  accel_z;
  float gyro_x,   gyro_y,   gyro_z;
  float linear_x, linear_y, linear_z;
  int   stability;
  unsigned long timestamp;
} latest;

enum SegState { RESTING, ACCUMULATING, IN_SEGMENT };
SegState      state              = RESTING;
int           seg_id             = 0;
int           active_frame_count = 0;           // frames above START_THRESHOLD
unsigned long seg_start_ms       = 0;
unsigned long last_active_ms     = 0;
unsigned long last_sample_ms     = 0;
bool          rotationReady      = false;

// Segment type determination: based on majority stability during segment
int           stable_frame_count = 0;           // count of frames with stability >= STABLE
int           transitioning_frame_count = 0;    // count of frames with stability == 2 (transitioning)
int           total_segment_frames = 0;         // total frames in current segment

// Position tracking: calculate hand position based on acceleration direction
float         accel_sum_x = 0, accel_sum_y = 0, accel_sum_z = 0;  // accumulated acceleration during transition
int           accel_frame_count = 0;            // frames counted for position calculation
char          current_position[32] = "NONE";    // discrete position: FRONT, RIGHT_UP, BACK_DOWN, etc.

// ─── Helpers ─────────────────────────────────────────────────────────────────
float gyroEnergy() {
  return sqrt(
    latest.gyro_x * latest.gyro_x +
    latest.gyro_y * latest.gyro_y +
    latest.gyro_z * latest.gyro_z
  );
}

char determineSegmentType() {
  // Determine if this segment was mostly STABLE or TRANSITIONING
  if (total_segment_frames == 0) return 'S';
  
  // If more than 60% of frames are marked "transitioning", label as T
  if (transitioning_frame_count > (total_segment_frames * 0.6)) {
    return 'T';
  }
  // Otherwise stable
  return 'S';
}

void calculatePosition() {
  // Convert accumulated acceleration to spherical coordinates
  // Reference: hand straight ahead (X+), sensor pointed up (Z+)
  // Spherical: azimuth (horizontal rotation), elevation (vertical angle)
  
  if (accel_frame_count == 0) {
    strcpy(current_position, "NONE");
    return;
  }
  
  // Average the acceleration
  float avg_x = accel_sum_x / accel_frame_count;
  float avg_y = accel_sum_y / accel_frame_count;
  float avg_z = accel_sum_z / accel_frame_count;
  
  // Calculate spherical coordinates
  // Azimuth: atan2(Y, X) — rotation in horizontal plane
  // Elevation: atan2(Z, sqrt(X²+Y²)) — angle from horizontal
  
  float azimuth = atan2(avg_y, avg_x) * 180.0 / PI;  // degrees, -180 to 180
  float elevation = atan2(avg_z, sqrt(avg_x*avg_x + avg_y*avg_y)) * 180.0 / PI;  // degrees, -90 to 90
  
  // Normalize azimuth to 0-360
  if (azimuth < 0) azimuth += 360;
  
  // Discretize to cardinal directions
  // Azimuth: 8 directions (45° each)
  //   0° = FRONT, 45° = FRONT_RIGHT, 90° = RIGHT, 135° = BACK_RIGHT,
  //   180° = BACK, 225° = BACK_LEFT, 270° = LEFT, 315° = FRONT_LEFT
  
  const char* azimuth_names[] = {"FRONT", "FRONT_RIGHT", "RIGHT", "BACK_RIGHT", 
                                  "BACK", "BACK_LEFT", "LEFT", "FRONT_LEFT"};
  int azimuth_index = (int)((azimuth + 22.5) / 45.0) % 8;
  
  // Elevation: 3 levels
  //   UP (> 30°), MIDDLE (-30° to 30°), DOWN (< -30°)
  const char* elevation_name;
  if (elevation > 30) {
    elevation_name = "UP";
  } else if (elevation < -30) {
    elevation_name = "DOWN";
  } else {
    elevation_name = "MIDDLE";
  }
  
  // Format: "FRONT_UP", "RIGHT_MIDDLE", "BACK_DOWN", etc.
  snprintf(current_position, sizeof(current_position), "%s_%s", 
           azimuth_names[azimuth_index], elevation_name);
}

void printHeader() {
  Serial.println(
    "seg_id,seg_type,position,timestamp,"
    "real,i,j,k,"
    "accel_x,accel_y,accel_z,"
    "gyro_x,gyro_y,gyro_z,"
    "linear_x,linear_y,linear_z,"
    "stability"
  );
}

void printFrame(char seg_type) {
  Serial.print(seg_id);             Serial.print(",");
  Serial.print(seg_type);           Serial.print(",");
  Serial.print(current_position);   Serial.print(",");
  Serial.print(latest.timestamp);   Serial.print(",");
  Serial.print(latest.real,     4); Serial.print(",");
  Serial.print(latest.i,        4); Serial.print(",");
  Serial.print(latest.j,        4); Serial.print(",");
  Serial.print(latest.k,        4); Serial.print(",");
  Serial.print(latest.accel_x,  4); Serial.print(",");
  Serial.print(latest.accel_y,  4); Serial.print(",");
  Serial.print(latest.accel_z,  4); Serial.print(",");
  Serial.print(latest.gyro_x,   4); Serial.print(",");
  Serial.print(latest.gyro_y,   4); Serial.print(",");
  Serial.print(latest.gyro_z,   4); Serial.print(",");
  Serial.print(latest.linear_x, 4); Serial.print(",");
  Serial.print(latest.linear_y, 4); Serial.print(",");
  Serial.print(latest.linear_z, 4); Serial.print(",");
  Serial.println(latest.stability);
}

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(3000);

  Wire.begin(D4, D5);

  if (!bno.begin_I2C()) {
    Serial.println("# ERROR: Failed to find BNO08x chip! Check wiring.");
    while (1);
  }

  bno.enableReport(SH2_ROTATION_VECTOR,      SAMPLE_INTERVAL_MS * 1000);
  bno.enableReport(SH2_ACCELEROMETER,        SAMPLE_INTERVAL_MS * 1000);
  bno.enableReport(SH2_GYROSCOPE_CALIBRATED, SAMPLE_INTERVAL_MS * 1000);
  bno.enableReport(SH2_LINEAR_ACCELERATION,  SAMPLE_INTERVAL_MS * 1000);
  bno.enableReport(SH2_STABILITY_CLASSIFIER, SAMPLE_INTERVAL_MS * 1000);

  printHeader();
  Serial.println("# Ready! Start signing. Ctrl+C when done.");
  Serial.print("# Thresholds — start: ");
  Serial.print(ENERGY_START_THRESHOLD);
  Serial.print("  end: ");
  Serial.print(ENERGY_END_THRESHOLD);
  Serial.print("  frames to start: ");
  Serial.print(FRAMES_TO_START);
  Serial.print("  min duration: ");
  Serial.print(MIN_SEGMENT_MS);
  Serial.println("ms");
}

// ─── Loop ────────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // Read all available sensor events
  if (bno.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        latest.real = sensorValue.un.rotationVector.real;
        latest.i    = sensorValue.un.rotationVector.i;
        latest.j    = sensorValue.un.rotationVector.j;
        latest.k    = sensorValue.un.rotationVector.k;
        rotationReady = true;
        break;
      case SH2_ACCELEROMETER:
        latest.accel_x = sensorValue.un.accelerometer.x;
        latest.accel_y = sensorValue.un.accelerometer.y;
        latest.accel_z = sensorValue.un.accelerometer.z;
        break;
      case SH2_GYROSCOPE_CALIBRATED:
        latest.gyro_x = sensorValue.un.gyroscope.x;
        latest.gyro_y = sensorValue.un.gyroscope.y;
        latest.gyro_z = sensorValue.un.gyroscope.z;
        break;
      case SH2_LINEAR_ACCELERATION:
        latest.linear_x = sensorValue.un.linearAcceleration.x;
        latest.linear_y = sensorValue.un.linearAcceleration.y;
        latest.linear_z = sensorValue.un.linearAcceleration.z;
        break;
      case SH2_STABILITY_CLASSIFIER:
        latest.stability = sensorValue.un.stabilityClassifier.classification;
        break;
    }
  }

  // Only proceed at target sample rate and when rotation vector is fresh
  if (!rotationReady || (now - last_sample_ms < SAMPLE_INTERVAL_MS)) return;
  rotationReady    = false;
  last_sample_ms   = now;
  latest.timestamp = now;

  float energy = gyroEnergy();

  // ── Segmentation state machine ────────────────────────────────────────────
  switch (state) {

    case RESTING:
      // Record rest frames with seg_id=0, type='R', position='NONE'
      strcpy(current_position, "NONE");
      printFrame('R');

      if (energy > ENERGY_START_THRESHOLD) {
        // Transition to ACCUMULATING: start building confidence
        active_frame_count = 1;
        state = ACCUMULATING;
        Serial.print("# Frame 1/");
        Serial.print(FRAMES_TO_START);
        Serial.println(" of movement detected...");
      }
      break;

    case ACCUMULATING:
      // Accumulate frames above START_THRESHOLD
      // Don't output yet — we're deciding if this is real motion
      
      if (energy > ENERGY_START_THRESHOLD) {
        active_frame_count++;
        
        if (active_frame_count >= FRAMES_TO_START) {
          // Sustained movement confirmed — start actual segment
          seg_id++;
          active_frame_count = 0;
          stable_frame_count = 0;
          transitioning_frame_count = 0;
          total_segment_frames = 0;
          accel_sum_x = 0;
          accel_sum_y = 0;
          accel_sum_z = 0;
          accel_frame_count = 0;
          strcpy(current_position, "NONE");
          seg_start_ms = now;
          last_active_ms = now;
          state = IN_SEGMENT;
          Serial.print("# Segment ");
          Serial.print(seg_id);
          Serial.println(" started — accumulating...");
        } else {
          Serial.print("# Frame ");
          Serial.print(active_frame_count);
          Serial.print("/");
          Serial.print(FRAMES_TO_START);
          Serial.println(" of movement...");
        }
      } else {
        // Movement stopped below START_THRESHOLD before reaching FRAMES_TO_START
        // Discard this accumulation and return to RESTING
        Serial.print("# Movement discarded (only ");
        Serial.print(active_frame_count);
        Serial.println(" frames) — returning to rest");
        active_frame_count = 0;
        state = RESTING;
      }
      break;

    case IN_SEGMENT:
      // Now actively recording a segment
      // Determine segment type based on stability
      if (latest.stability == 2) {
        transitioning_frame_count++;
        // Accumulate acceleration during transitioning
        accel_sum_x += latest.linear_x;
        accel_sum_y += latest.linear_y;
        accel_sum_z += latest.linear_z;
        accel_frame_count++;
      } else if (latest.stability >= STABILITY_THRESHOLD) {
        stable_frame_count++;
      }
      total_segment_frames++;

      // Output frame with placeholder type '?' (will be finalized on segment end)
      printFrame('?');

      if (energy > ENERGY_END_THRESHOLD) {
        last_active_ms = now;
      }

      // End segment after sufficient quiet period
      if ((now - last_active_ms) > DEBOUNCE_MS) {
        unsigned long duration = now - seg_start_ms;

        if (duration < MIN_SEGMENT_MS) {
          // Discard: segment too short
          Serial.print("# Segment ");
          Serial.print(seg_id);
          Serial.println(" DISCARDED (too short)");
          // Note: frames already printed — Python will filter these out
          seg_id--;
        } else {
          // Calculate final position based on acceleration
          calculatePosition();
          
          // Finalize segment type
          char final_type = determineSegmentType();
          Serial.print("# Segment ");
          Serial.print(seg_id);
          Serial.print(" ended (");
          Serial.print(duration);
          Serial.print("ms) — type: ");
          Serial.print(final_type);
          Serial.print(" position: ");
          Serial.println(current_position);
        }
        
        state = RESTING;
        active_frame_count = 0;
        strcpy(current_position, "NONE");
        accel_sum_x = 0;
        accel_sum_y = 0;
        accel_sum_z = 0;
        accel_frame_count = 0;
      }
      break;
  }
}
