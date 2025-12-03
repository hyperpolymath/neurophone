package ai.neurophone

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.json.JSONObject

/**
 * Main activity for the NeuroSymbolic Phone application.
 * Manages sensor input, neural processing, and LLM interaction.
 */
class MainActivity : AppCompatActivity(), SensorEventListener {

    companion object {
        private const val TAG = "NeuroPhone"
        private const val PERMISSION_REQUEST_CODE = 100
    }

    // UI Components
    private lateinit var statusText: TextView
    private lateinit var neuralContextText: TextView
    private lateinit var inputField: EditText
    private lateinit var responseText: TextView
    private lateinit var startButton: Button
    private lateinit var sendButton: Button
    private lateinit var localSwitch: Switch
    private lateinit var activityIndicator: ProgressBar

    // Sensor management
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var magnetometer: Sensor? = null
    private var lightSensor: Sensor? = null
    private var proximitySensor: Sensor? = null

    // Coroutine scope for background tasks
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // State
    private var isSystemRunning = false
    private var contextUpdateJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initializeViews()
        initializeSensors()
        initializeSystem()

        checkPermissions()
    }

    private fun initializeViews() {
        statusText = findViewById(R.id.statusText)
        neuralContextText = findViewById(R.id.neuralContextText)
        inputField = findViewById(R.id.inputField)
        responseText = findViewById(R.id.responseText)
        startButton = findViewById(R.id.startButton)
        sendButton = findViewById(R.id.sendButton)
        localSwitch = findViewById(R.id.localSwitch)
        activityIndicator = findViewById(R.id.activityIndicator)

        startButton.setOnClickListener { toggleSystem() }
        sendButton.setOnClickListener { sendQuery() }

        updateUI()
    }

    private fun initializeSensors() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
        proximitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY)

        Log.d(TAG, "Sensors initialized: " +
            "accel=${accelerometer != null}, " +
            "gyro=${gyroscope != null}, " +
            "mag=${magnetometer != null}, " +
            "light=${lightSensor != null}, " +
            "prox=${proximitySensor != null}")
    }

    private fun initializeSystem() {
        scope.launch(Dispatchers.IO) {
            try {
                val config = createConfig()
                val success = NativeLib.init(config)

                withContext(Dispatchers.Main) {
                    if (success) {
                        statusText.text = "System initialized"
                        Log.i(TAG, "Native system initialized")
                    } else {
                        statusText.text = "Initialization failed"
                        Log.e(TAG, "Failed to initialize native system")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing system", e)
                withContext(Dispatchers.Main) {
                    statusText.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun createConfig(): String {
        return JSONObject().apply {
            put("loop_interval_ms", 20)
            put("debug", false)
            // Sensor config
            put("sensor", JSONObject().apply {
                put("sample_rate_hz", 50.0)
                put("buffer_size", 100)
                put("output_dim", 32)
            })
            // LSM config (optimized for Oppo Reno 13)
            put("lsm", JSONObject().apply {
                put("dimensions", listOf(8, 8, 8))
                put("spectral_radius", 0.9)
            })
            // ESN config
            put("esn", JSONObject().apply {
                put("reservoir_size", 300)
                put("spectral_radius", 0.95)
            })
        }.toString()
    }

    private fun toggleSystem() {
        if (isSystemRunning) {
            stopSystem()
        } else {
            startSystem()
        }
    }

    private fun startSystem() {
        scope.launch(Dispatchers.IO) {
            val success = NativeLib.start()

            withContext(Dispatchers.Main) {
                if (success) {
                    isSystemRunning = true
                    registerSensors()
                    startContextUpdates()
                    updateUI()
                    statusText.text = "System running"
                } else {
                    statusText.text = "Failed to start"
                }
            }
        }
    }

    private fun stopSystem() {
        scope.launch(Dispatchers.IO) {
            NativeLib.stop()

            withContext(Dispatchers.Main) {
                isSystemRunning = false
                unregisterSensors()
                stopContextUpdates()
                updateUI()
                statusText.text = "System stopped"
            }
        }
    }

    private fun registerSensors() {
        val samplingPeriod = SensorManager.SENSOR_DELAY_GAME // ~20ms

        accelerometer?.let { sensorManager.registerListener(this, it, samplingPeriod) }
        gyroscope?.let { sensorManager.registerListener(this, it, samplingPeriod) }
        magnetometer?.let { sensorManager.registerListener(this, it, samplingPeriod) }
        lightSensor?.let { sensorManager.registerListener(this, it, samplingPeriod) }
        proximitySensor?.let { sensorManager.registerListener(this, it, samplingPeriod) }

        Log.d(TAG, "Sensors registered")
    }

    private fun unregisterSensors() {
        sensorManager.unregisterListener(this)
        Log.d(TAG, "Sensors unregistered")
    }

    private fun startContextUpdates() {
        contextUpdateJob = scope.launch {
            while (isActive && isSystemRunning) {
                try {
                    val context = withContext(Dispatchers.IO) {
                        NativeLib.getNeuralContext()
                    }
                    neuralContextText.text = context
                } catch (e: Exception) {
                    Log.w(TAG, "Error updating context", e)
                }
                delay(500) // Update every 500ms
            }
        }
    }

    private fun stopContextUpdates() {
        contextUpdateJob?.cancel()
        contextUpdateJob = null
    }

    private fun sendQuery() {
        val message = inputField.text.toString().trim()
        if (message.isEmpty()) {
            Toast.makeText(this, "Please enter a message", Toast.LENGTH_SHORT).show()
            return
        }

        activityIndicator.visibility = View.VISIBLE
        sendButton.isEnabled = false

        scope.launch {
            try {
                val preferLocal = localSwitch.isChecked
                val response = withContext(Dispatchers.IO) {
                    NativeLib.query(message, preferLocal)
                }

                responseText.text = response.ifEmpty { "No response received" }
            } catch (e: Exception) {
                Log.e(TAG, "Query error", e)
                responseText.text = "Error: ${e.message}"
            } finally {
                activityIndicator.visibility = View.GONE
                sendButton.isEnabled = true
            }
        }
    }

    private fun updateUI() {
        startButton.text = if (isSystemRunning) "Stop" else "Start"
        sendButton.isEnabled = isSystemRunning
        inputField.isEnabled = isSystemRunning
    }

    // SensorEventListener implementation
    override fun onSensorChanged(event: SensorEvent) {
        if (!isSystemRunning) return

        // Send sensor data to native system
        scope.launch(Dispatchers.IO) {
            NativeLib.processSensor(
                event.sensor.type,
                event.values.clone(),
                event.timestamp,
                event.accuracy
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        Log.d(TAG, "Sensor ${sensor.name} accuracy changed to $accuracy")
    }

    // Permissions
    private fun checkPermissions() {
        val permissions = mutableListOf<String>()

        // Body sensors permission (for heart rate, etc. if available)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.BODY_SENSORS)
        }

        // Internet permission for Claude API
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.INTERNET)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.INTERNET)
        }

        if (permissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissions.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSION_REQUEST_CODE) {
            val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (!allGranted) {
                Toast.makeText(this, "Some permissions denied", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (isSystemRunning) {
            registerSensors()
            startContextUpdates()
        }
    }

    override fun onPause() {
        super.onPause()
        unregisterSensors()
        stopContextUpdates()
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        if (isSystemRunning) {
            NativeLib.stop()
        }
    }
}
