package org.tensorflow.lite.examples.posenet
import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.Fragment
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import org.tensorflow.lite.examples.posenet.lib.BodyPart
import org.tensorflow.lite.examples.posenet.lib.Person
import org.tensorflow.lite.examples.posenet.lib.Posenet
import kotlin.math.*
import kotlin.system.measureNanoTime

class Planks:
Fragment(),
ActivityCompat.OnRequestPermissionsResultCallback {
    private val bodyJoints = listOf(
        Pair(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
        Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
        Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
        Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
        Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
        Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
        Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
        Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
        Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
        Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
        Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
        Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
    )

    /** Threshold for confidence score. */
    private val minConfidence = 0.6

    // time elapsed
    private var timeStart = System.currentTimeMillis()
    private var timeElapsed = System.currentTimeMillis()

    /** Radius of circle used to draw keypoints.  */
    private val circleRadius = 9.0f

    /** Paint class holds the style and color information to draw geometries,text and bitmaps. */
    private var kpPaint = Paint()
    // paint class instance for pose lines
    private var linePaint = Paint()
    // paint class instance for text
    private var textPaint = Paint()
    // for writing coordinates of each detected keypoint
    private var bodyPartCoordTextPaint = Paint()
    // for plank text
    private var plankTextPaint = Paint()
    // for squat text
    private var squatTextPaint = Paint()

    //accumulated plank time
    private var plank_begin = System.currentTimeMillis()
    var plank_time_elapsed = System.currentTimeMillis()
    var lh_le_ls_angle: Int = 0
    var rh_re_rs_angle: Int=0
    var ls_lhip_la_angle: Int=0
    var rs_rhip_ra_angle: Int=0

    /** A shape for extracting frame data.   */
    private val PREVIEW_WIDTH = 320
    private val PREVIEW_HEIGHT = 240

    /** An object for the Posenet library.    */
    private lateinit var posenet: Posenet

    /** ID of the current [CameraDevice].   */
    private var cameraId: String? = null

    /** A [SurfaceView] for camera preview.   */
    private var surfaceView: SurfaceView? = null

    /** A [CameraCaptureSession] for camera preview.   */
    private var captureSession: CameraCaptureSession? = null

    /** A reference to the opened [CameraDevice].    */
    private var cameraDevice: CameraDevice? = null

    /** The [android.util.Size] of camera preview.  */
    private var previewSize: Size? = null

    /** The [android.util.Size.getWidth] of camera preview. */
    private var previewWidth = 0

    /** The [android.util.Size.getHeight] of camera preview.  */
    private var previewHeight = 0

    /** A counter to keep count of total frames before performing inference.  */
    private var frameCounter = 0

    // frame counter to keep track of total number of frames, to track fps
    private var numFrames = 0

    // inference frame rate; how many frames before performing inference
    private var inferenceRate = 3

    // for fps
    private var fps = 0

    /** An IntArray to save image data in ARGB8888 format  */
    private lateinit var rgbBytes: IntArray

    /** A ByteArray to save image data in YUV format  */
    private var yuvBytes = arrayOfNulls<ByteArray>(3)

    /** An additional thread for running tasks that shouldn't block the UI.   */
    private var backgroundThread: HandlerThread? = null

    /** A [Handler] for running tasks in the background.    */
    private var backgroundHandler: Handler? = null

    /** An [ImageReader] that handles preview frame capture.   */
    private var imageReader: ImageReader? = null

    /** [CaptureRequest.Builder] for the camera preview   */
    private var previewRequestBuilder: CaptureRequest.Builder? = null

    /** [CaptureRequest] generated by [.previewRequestBuilder   */
    private var previewRequest: CaptureRequest? = null

    /** A [Semaphore] to prevent the app from exiting before closing the camera.    */
    private val cameraOpenCloseLock = Semaphore(1)

    /** Whether the current camera device supports Flash or not.    */
    private var flashSupported = false

    /** Orientation of the camera sensor.   */
    private var sensorOrientation: Int? = null

    /** Abstract interface to someone holding a display surface.    */
    private var surfaceHolder: SurfaceHolder? = null

    /** [CameraDevice.StateCallback] is called when [CameraDevice] changes its state.   */
    private val stateCallback = object : CameraDevice.StateCallback() {

        override fun onOpened(cameraDevice: CameraDevice) {
            cameraOpenCloseLock.release()
            this@Planks.cameraDevice = cameraDevice
            createCameraPreviewSession()
        }

        override fun onDisconnected(cameraDevice: CameraDevice) {
            cameraOpenCloseLock.release()
            cameraDevice.close()
            this@Planks.cameraDevice = null
        }

        override fun onError(cameraDevice: CameraDevice, error: Int) {
            onDisconnected(cameraDevice)
            this@Planks.activity?.finish()
        }
    }

    /**
     * A [CameraCaptureSession.CaptureCallback] that handles events related to JPEG capture.
     */
    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureProgressed(
            session: CameraCaptureSession,
            request: CaptureRequest,
            partialResult: CaptureResult
        ) {
        }

        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
        }
    }

    /**
     * Shows a [Toast] on the UI thread.
     *
     * @param text The message to show
     */
    private fun showToast(text: String) {
        val activity = activity
        activity?.runOnUiThread { Toast.makeText(activity, text, Toast.LENGTH_SHORT).show() }
    }


    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? = inflater.inflate(R.layout.tfe_pn_activity_posenet, container, false)
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        surfaceView = view.findViewById(R.id.surfaceView)
        surfaceHolder = surfaceView!!.holder
        super.onViewCreated(view, savedInstanceState)

        //change activity from a button
        val botonir = view.findViewById(R.id.bt) as Button
        botonir.setOnClickListener{
            var time:String = plank_time_elapsed.toString()
            var lh_le_ls_angle:Int = lh_le_ls_angle
            var rh_re_rs_angle:Int = rh_re_rs_angle
            var ls_lhip_la_angle:Int = ls_lhip_la_angle
            var rs_rhip_ra_angle:Int = rs_rhip_ra_angle
            val intent = Intent(context, CameraActivity::class.java)
            intent.putExtra("time", time)
            intent.putExtra("lh_le_ls_angle", lh_le_ls_angle)
            intent.putExtra("rh_re_rs_angle", rh_re_rs_angle)
            intent.putExtra("ls_lhip_la_angle", ls_lhip_la_angle)
            intent.putExtra("rs_rhip_ra_angle", rs_rhip_ra_angle)

            startActivity(intent)

        }
    }


    override fun onResume() {
        super.onResume()
        startBackgroundThread()
    }

    override fun onStart() {
        super.onStart()
        openCamera()
        posenet = Posenet(this.context!!)
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        posenet.close()
    }

    private fun requestCameraPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            ConfirmationDialog().show(childFragmentManager, FRAGMENT_DIALOG)
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (allPermissionsGranted(grantResults)) {
                ErrorDialog.newInstance(getString(R.string.tfe_pn_request_permission))
                    .show(childFragmentManager, FRAGMENT_DIALOG)
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    }

    private fun allPermissionsGranted(grantResults: IntArray) = grantResults.all {
        it == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Sets up member variables related to camera.
     */
    private fun setUpCameraOutputs() {

        val activity = activity
        val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (cameraId in manager.cameraIdList) {
                val characteristics = manager.getCameraCharacteristics(cameraId)

                // We don't use a front facing camera in this sample.
                val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (cameraDirection != null &&
                    cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
                ) {
                    continue
                }

                previewSize = Size(PREVIEW_WIDTH, PREVIEW_HEIGHT)

                imageReader = ImageReader.newInstance(
                    PREVIEW_WIDTH, PREVIEW_HEIGHT,
                    ImageFormat.YUV_420_888, /*maxImages*/ 2
                )

                sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!

                previewHeight = previewSize!!.height
                previewWidth = previewSize!!.width

                // Initialize the storage bitmaps once when the resolution is known.
                rgbBytes = IntArray(previewWidth * previewHeight)

                // Check if the flash is supported.
                flashSupported =
                    characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true

                this.cameraId = cameraId

                // We've found a viable camera and finished setting up member variables,
                // so we don't need to iterate through other available cameras.
                return
            }
        } catch (e: CameraAccessException) {
            Log.e(TAG, e.toString())
        } catch (e: NullPointerException) {
            // Currently an NPE is thrown when the Camera2API is used but not supported on the
            // device this code runs.
            ErrorDialog.newInstance(getString(R.string.tfe_pn_camera_error))
                .show(childFragmentManager, FRAGMENT_DIALOG)
        }
    }

    /**
     * Opens the camera specified by [PosenetActivity.cameraId].
     */
    private fun openCamera() {
        val permissionCamera = ContextCompat.checkSelfPermission(activity!!, Manifest.permission.CAMERA)
        if (permissionCamera != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission()
        }
        setUpCameraOutputs()
        val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            // Wait for camera to open - 2.5 seconds is sufficient
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw RuntimeException("Time out waiting to lock camera opening.")
            }
            manager.openCamera(cameraId!!, stateCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, e.toString())
        } catch (e: InterruptedException) {
            throw RuntimeException("Interrupted while trying to lock camera opening.", e)
        }
    }

    /**
     * Closes the current [CameraDevice].
     */
    private fun closeCamera() {
        if (captureSession == null) {
            return
        }

        try {
            cameraOpenCloseLock.acquire()
            captureSession!!.close()
            captureSession = null
            cameraDevice!!.close()
            cameraDevice = null
            imageReader!!.close()
            imageReader = null
        } catch (e: InterruptedException) {
            throw RuntimeException("Interrupted while trying to lock camera closing.", e)
        } finally {
            cameraOpenCloseLock.release()
        }
    }

    /**
     * Starts a background thread and its [Handler].
     */
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("imageAvailableListener").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    /**
     * Stops the background thread and its [Handler].
     */
    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, e.toString())
        }
    }

    /** Fill the yuvBytes with data from image planes.   */
    private fun fillBytes(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray?>) {
        // Row stride is the total number of bytes occupied in memory by a row of an image.
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer.get(yuvBytes[i]!!)
        }
    }

    /** A [OnImageAvailableListener] to receive frames as they are available.  */
    private var imageAvailableListener = object : OnImageAvailableListener {
        override fun onImageAvailable(imageReader: ImageReader) {
            // We need wait until we have some size from onPreviewSizeChosen
            if (previewWidth == 0 || previewHeight == 0) {
                return
            }

            val image = imageReader.acquireLatestImage() ?: return
            fillBytes(image.planes, yuvBytes)

            ImageUtils.convertYUV420ToARGB8888(
                yuvBytes[0]!!,
                yuvBytes[1]!!,
                yuvBytes[2]!!,
                previewWidth,
                previewHeight,
                /*yRowStride=*/ image.planes[0].rowStride,
                /*uvRowStride=*/ image.planes[1].rowStride,
                /*uvPixelStride=*/ image.planes[1].pixelStride,
                rgbBytes
            )

            // Create bitmap from int array
            val imageBitmap = Bitmap.createBitmap(
                rgbBytes, previewWidth, previewHeight,
                Bitmap.Config.ARGB_8888
            )

            // Create rotated version for portrait display
            val rotateMatrix = Matrix()
            rotateMatrix.postRotate(90.0f)

            val rotatedBitmap = Bitmap.createBitmap(
                imageBitmap, 0, 0, previewWidth, previewHeight,
                rotateMatrix, true
            )
            image.close()

            // Process an image for analysis at a specified frame rate
            frameCounter =(frameCounter + 1) % 3
            if (frameCounter == 0) {
                processImage(rotatedBitmap)
            }
            //processImage(rotatedBitmap)

            // update number of frames
            numFrames += 1

        }
    }

    /** Crop Bitmap to maintain aspect ratio of model input.   */
    private fun cropBitmap(bitmap: Bitmap): Bitmap {
        val bitmapRatio = bitmap.height.toFloat() / bitmap.width
        val modelInputRatio = MODEL_HEIGHT.toFloat() / MODEL_WIDTH
        var croppedBitmap = bitmap

        // Acceptable difference between the modelInputRatio and bitmapRatio to skip cropping.
        val maxDifference = 1e-5

        // Checks if the bitmap has similar aspect ratio as the required model input.
        when {
            abs(modelInputRatio - bitmapRatio) < maxDifference -> return croppedBitmap
            modelInputRatio < bitmapRatio -> {
                // New image is taller so we are height constrained.
                val cropHeight = bitmap.height - (bitmap.width.toFloat() / modelInputRatio)
                croppedBitmap = Bitmap.createBitmap(
                    bitmap,
                    0,
                    (cropHeight / 2).toInt(),
                    bitmap.width,
                    (bitmap.height - cropHeight).toInt()
                )
            }
            else -> {
                val cropWidth = bitmap.width - (bitmap.height.toFloat() * modelInputRatio)
                croppedBitmap = Bitmap.createBitmap(
                    bitmap,
                    (cropWidth / 2).toInt(),
                    0,
                    (bitmap.width - cropWidth).toInt(),
                    bitmap.height
                )
            }
        }
        return croppedBitmap
    }

    /** Set the paint color and size.    */
    private fun setTextPaint() {
        textPaint.color = Color.GREEN
        textPaint.textSize = 60.0f
        textPaint.strokeWidth = 8.0f
    }

    private fun setPlankTextPaint(){
        plankTextPaint.color = Color.RED
        plankTextPaint.textSize = 40.0f
    }

    private fun setSquatTextPaint(){
        squatTextPaint.color = Color.rgb(24, 172, 228)
        squatTextPaint.textSize = 40.0f
    }

    private fun setKpPaint(){
        kpPaint.color = Color.rgb(230, 15, 195)
        kpPaint.strokeWidth = 8.0f
    }

    private fun setPoseLinePaint(){
        linePaint.color = Color.rgb(20, 237, 220)
        linePaint.strokeWidth = 8.0f
    }

    private fun setBodyPartCoordPaint(){
        bodyPartCoordTextPaint.color = Color.rgb(50, 186, 3)
        bodyPartCoordTextPaint.textSize = 40.0f
        bodyPartCoordTextPaint.strokeWidth = 20.0f
    }


    private fun find_bodypart_xy(person: Person, part_idx: Int, widthRatio: Float, heightRatio: Float,
                                 left: Int, top: Int): List<Float>{
        var x_y = listOf<Float>(0.0f, 0.0f)
        try {
            var bodyPart = person.keyPoints[part_idx]
            var position = bodyPart.position
            var bodyPartX: Float = position.x.toFloat() * widthRatio + left
            var bodyPartY: Float = position.y.toFloat() * heightRatio + top

            x_y = listOf(bodyPartX, bodyPartY)

        }catch (e: Exception){
            Log.i(
                "bodyPart",
                "Cannot find body part"
            )
        }
        return x_y
    }


    private fun get_angle(pt0: List<Float>, pt1: List<Float>, pt2: List<Float>): Int {
        /**
         * pt0: reference point from which we measure the angle between pt1 and pt2
         * pt1: point 1 of interest
         * pt2: point 2 of interest
         * return: angle between pt0-pt1, and pt0-pt2
         */
        try {
            // cosine rule
            var a2 = (pt2[0] - pt1[0]).pow(2) + (pt2[1] - pt1[1]).pow(2)
            var b2 = (pt2[0] - pt0[0]).pow(2) + (pt2[1] - pt0[1]).pow(2)
            var c2 = (pt1[0] - pt0[0]).pow(2) + (pt1[1] - pt0[1]).pow(2)

            var angle_deg = (acos((b2 + c2 - a2) / sqrt(4 * b2 * c2)) * 180.0/PI).toInt()

            return angle_deg

        } catch (e: Exception){
            return 0
        }
    }

    // function to analyze planks
    private fun analyze_plank(canvas: Canvas, person: Person){
        /**
         * lh_le_ls_angle: angle between left hand, left elbow and left elbow/shoulder
         * rh_re_rs_angle: angle between right hand, right elbow and right elbow/shoulder
         * ll_lhip_la_angle: angle between left shoulder, left hip and left ankle
         * rl_rhip_ra: angle between right shoulder, right hip and right ankle
         */

        val screenWidth: Int
        val screenHeight: Int
        val left: Int
        val top: Int

        if (canvas.height > canvas.width) {
            screenWidth = canvas.width
            screenHeight = canvas.width
            left = 0
            top = (canvas.height - canvas.width) / 2
        } else {
            screenWidth = canvas.height
            screenHeight = canvas.height
            left = (canvas.width - canvas.height) / 2
            top = 0
        }

        val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
        val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT


        var arm_angle_range = 60..120
        var hip_angle_range = 140..180


        lh_le_ls_angle = get_angle(find_bodypart_xy(person, 7, widthRatio, heightRatio, left, top),
            find_bodypart_xy(person, 5, widthRatio, heightRatio, left, top),
            find_bodypart_xy(person, 9, widthRatio, heightRatio, left, top))

        rh_re_rs_angle = get_angle(find_bodypart_xy(person, 8, widthRatio, heightRatio, left, top),
            find_bodypart_xy(person, 6, widthRatio, heightRatio, left, top),
            find_bodypart_xy(person, 10, widthRatio, heightRatio, left, top))

        ls_lhip_la_angle = get_angle(find_bodypart_xy(person, 11, widthRatio, heightRatio, left, top),
            find_bodypart_xy(person, 5, widthRatio, heightRatio, left, top),
            find_bodypart_xy(person, 15, widthRatio, heightRatio, left, top))

        rs_rhip_ra_angle = get_angle(find_bodypart_xy(person, 12, widthRatio, heightRatio, left, top),
            find_bodypart_xy(person, 6, widthRatio, heightRatio, left, top),
            find_bodypart_xy(person, 16, widthRatio, heightRatio, left, top))


        canvas.drawText(
            "Left Arm angle: %d".format(lh_le_ls_angle),
            (15.0f * widthRatio),
            (-40.0f * heightRatio + top),
            bodyPartCoordTextPaint
        )

        canvas.drawText(
            "Right Arm angle: %d".format(rh_re_rs_angle),
            (15.0f * widthRatio),
            (-30.0f * heightRatio + top),
            bodyPartCoordTextPaint
        )

        canvas.drawText(
            "Left Hip Angle: %d".format(ls_lhip_la_angle),
            (15.0f * widthRatio),
            (-20.0f * heightRatio + top),
            bodyPartCoordTextPaint
        )

        canvas.drawText(
            "Right Hip Angle: %d".format(rs_rhip_ra_angle),
            (15.0f * widthRatio),
            (-10.0f * heightRatio + top),
            bodyPartCoordTextPaint

        )

        // if angle conditions are met, flash total duration under tension
        if((lh_le_ls_angle in arm_angle_range || rh_re_rs_angle in arm_angle_range)
            && (ls_lhip_la_angle in hip_angle_range || rs_rhip_ra_angle in hip_angle_range)
        //&& (n_la_dist in la_ra_dist_range || n_ra_dist in la_ra_dist_range)
        ){

            plank_time_elapsed = System.currentTimeMillis() - plank_begin

            canvas.drawText(
                "Plank",
                (120.0f * widthRatio),
                (-20.0f * heightRatio + top),
                plankTextPaint
            )

            canvas.drawText(
                "Time elapsed: %.2f s".format(plank_time_elapsed * 1.0f / 1000),
                (120.0f * widthRatio),
                (-40.0f * heightRatio + top),
                plankTextPaint
            )

        }
    }


    /** Draw bitmap on Canvas.   */
    private fun draw(canvas: Canvas, person: Person, bitmap: Bitmap) {
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        // Draw `bitmap` and `person` in square canvas.
        val screenWidth: Int
        val screenHeight: Int
        val left: Int
        val right: Int
        val top: Int
        val bottom: Int
        if (canvas.height > canvas.width) {
            screenWidth = canvas.width
            screenHeight = canvas.width
            left = 0
            top = (canvas.height - canvas.width) / 2
        } else {
            screenWidth = canvas.height
            screenHeight = canvas.height
            left = (canvas.width - canvas.height) / 2
            top = 0
        }
        right = left + screenWidth
        bottom = top + screenHeight

        // initialize the paint settings
        setTextPaint()
        setKpPaint()
        setPoseLinePaint()
        setBodyPartCoordPaint()
        setPlankTextPaint()
        setSquatTextPaint()

        canvas.drawBitmap(
            bitmap,
            Rect(0, 0, bitmap.width, bitmap.height),
            Rect(left, top, right, bottom),
            textPaint
        )

        val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
        val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

        // Draw key points over the image.
        for (keyPoint in person.keyPoints) {
            if (keyPoint.score > minConfidence) {
                val position = keyPoint.position
                val adjustedX: Float = position.x.toFloat() * widthRatio + left
                val adjustedY: Float = position.y.toFloat() * heightRatio + top
                canvas.drawCircle(adjustedX, adjustedY, circleRadius, kpPaint)
            }
        }

        for (line in bodyJoints) {
            if (
                (person.keyPoints[line.first.ordinal].score > minConfidence) and
                (person.keyPoints[line.second.ordinal].score > minConfidence)
            ) {
                canvas.drawLine(
                    person.keyPoints[line.first.ordinal].position.x.toFloat() * widthRatio + left,
                    person.keyPoints[line.first.ordinal].position.y.toFloat() * heightRatio + top,
                    person.keyPoints[line.second.ordinal].position.x.toFloat() * widthRatio + left,
                    person.keyPoints[line.second.ordinal].position.y.toFloat() * heightRatio + top,
                    linePaint
                )
            }
        }

        if(person.score > minConfidence){
            // PLANK ANALYZER
            analyze_plank(canvas, person)
        }

        //** Stuff below the camera's image **//
        canvas.drawText(
            "Score: %.2f".format(person.score),
            (15.0f * widthRatio),
            (20.0f * heightRatio + bottom),
            textPaint
        )
        canvas.drawText(
            "Device: %s".format(posenet.device),
            (15.0f * widthRatio),
            (40.0f * heightRatio + bottom),
            textPaint
        )
        canvas.drawText(
            "Time Elapsed: %.2f s".format(timeElapsed * 1.0f / 1_000),
            (15.0f * widthRatio),
            (60.0f * heightRatio + bottom),
            textPaint
        )

        canvas.drawText(
            "FPS: %.2f".format(numFrames / (timeElapsed * 1.0f / 1_000)),
            (15.0f * widthRatio),
            (80.0f * heightRatio + bottom),
            textPaint
        )

        // Draw!
        surfaceHolder!!.unlockCanvasAndPost(canvas)
    }


    /** Process image using Posenet library.   */
    private fun processImage(bitmap: Bitmap) {
        // Crop bitmap.
        val croppedBitmap = cropBitmap(bitmap)

        // Created scaled version of bitmap for model input.
        val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_WIDTH, MODEL_HEIGHT, true)

        // Perform inference.
        val person = posenet.estimateSinglePose(scaledBitmap)
        val canvas: Canvas = surfaceHolder!!.lockCanvas()
        draw(canvas, person, scaledBitmap)

        timeElapsed = System.currentTimeMillis() - timeStart
    }

    /**
     * Creates a new [CameraCaptureSession] for camera preview.
     */
    private fun createCameraPreviewSession() {
        try {

            // We capture images from preview in YUV format.
            imageReader = ImageReader.newInstance(
                previewSize!!.width, previewSize!!.height, ImageFormat.YUV_420_888, 2
            )
            imageReader!!.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)

            // This is the surface we need to record images for processing.
            val recordingSurface = imageReader!!.surface

            // We set up a CaptureRequest.Builder with the output Surface.
            previewRequestBuilder = cameraDevice!!.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            )
            previewRequestBuilder!!.addTarget(recordingSurface)

            // Here, we create a CameraCaptureSession for camera preview.
            cameraDevice!!.createCaptureSession(
                listOf(recordingSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
                        // The camera is already closed
                        if (cameraDevice == null) return

                        // When the session is ready, we start displaying the preview.
                        captureSession = cameraCaptureSession
                        try {
                            // Auto focus should be continuous for camera preview.
                            previewRequestBuilder!!.set(
                                CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                            )
                            // Flash is automatically enabled when necessary.
                            setAutoFlash(previewRequestBuilder!!)

                            // Finally, we start displaying the camera preview.
                            previewRequest = previewRequestBuilder!!.build()
                            captureSession!!.setRepeatingRequest(
                                previewRequest!!,
                                captureCallback, backgroundHandler
                            )
                        } catch (e: CameraAccessException) {
                            Log.e(TAG, e.toString())
                        }
                    }

                    override fun onConfigureFailed(cameraCaptureSession: CameraCaptureSession) {
                        showToast("Failed")
                    }
                },
                null
            )
        } catch (e: CameraAccessException) {
            Log.e(TAG, e.toString())
        }
    }

    private fun setAutoFlash(requestBuilder: CaptureRequest.Builder) {
        if (flashSupported) {
            requestBuilder.set(
                CaptureRequest.CONTROL_AE_MODE,
                CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
            )
        }
    }

    /**
     * Shows an error message dialog.
     */
    class ErrorDialog : DialogFragment() {

        override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
            AlertDialog.Builder(activity)
                .setMessage(arguments!!.getString(ARG_MESSAGE))
                .setPositiveButton(android.R.string.ok) { _, _ -> activity!!.finish() }
                .create()


        companion object {

            @JvmStatic
            private val ARG_MESSAGE = "message"

            @JvmStatic
            fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
                arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
            }
        }
    }

    companion object {
        /**
         * Conversion from screen rotation to JPEG orientation.
         */
        private val ORIENTATIONS = SparseIntArray()
        private val FRAGMENT_DIALOG = "dialog"

        init {
            ORIENTATIONS.append(Surface.ROTATION_0, 90)
            ORIENTATIONS.append(Surface.ROTATION_90, 0)
            ORIENTATIONS.append(Surface.ROTATION_180, 270)
            ORIENTATIONS.append(Surface.ROTATION_270, 180)
        }

        /**
         * Tag for the [Log].
         */
        private const val TAG = "PosenetActivity"
    }
}