package com.example.mediscanai

import android.content.Context
import android.content.Intent
import android.graphics.*
import android.graphics.pdf.PdfDocument
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import androidx.navigation.NavController
import androidx.navigation.compose.*
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class Report(val type: String, val result: String, val time: String)

class MainActivity : ComponentActivity() {

    private var fractureInterpreter: Interpreter? = null
    private var pneumoniaInterpreter: Interpreter? = null
    private var mriInterpreter: Interpreter? = null
    private var ctInterpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Models Load Logic with Crash Protection
        try {
            fractureInterpreter = Interpreter(loadModelFile("fracture_model.tflite"))
            pneumoniaInterpreter = Interpreter(loadModelFile("pneumonia_model.tflite"))
            mriInterpreter = Interpreter(loadModelFile("mri_model.tflite"))
            ctInterpreter = Interpreter(loadModelFile("ct_model.tflite"))
        } catch (e: Exception) {
            e.printStackTrace()
        }

        setContent {
            val darkTheme = isSystemInDarkTheme()
            MaterialTheme(colorScheme = if (darkTheme) darkColorScheme() else lightColorScheme()) {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    AppNavigator(fractureInterpreter, mriInterpreter, ctInterpreter, pneumoniaInterpreter)
                }
            }
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }
}

@Composable
fun AppNavigator(f: Interpreter?, m: Interpreter?, c: Interpreter?, p: Interpreter?) {
    val navController = rememberNavController()
    val historyList = remember { mutableStateListOf<Report>() }

    NavHost(navController, startDestination = "home") {
        composable("home") { HomeScreen(navController) }
        composable("fracture") { f?.let { UniversalScanScreen("Fracture", it, historyList, navController) } }
        composable("mri") { m?.let { UniversalScanScreen("MRI", it, historyList, navController) } }
        composable("ct") { c?.let { UniversalScanScreen("CT Scan", it, historyList, navController) } }
        composable("pneumonia") { p?.let { UniversalScanScreen("Pneumonia", it, historyList, navController) } }
        composable("blood") { BloodReportSection(navController) }
        composable("chat") { ChatScreen(navController) }
    }
}

@Composable
fun HomeScreen(navController: NavController) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("MediScan AI 🚀", fontSize = 32.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
        Text("Advanced Diagnostic Assistant", fontSize = 16.sp, color = MaterialTheme.colorScheme.secondary)

        Spacer(modifier = Modifier.height(20.dp))

        val menuItems = listOf(
            "Fracture Detection" to "fracture",
            "MRI Brain Analysis" to "mri",
            "CT Lung Scan" to "ct",
            "Pneumonia Check" to "pneumonia",
            "Blood Report Analysis" to "blood"
        )

        menuItems.forEach { (label, route) ->
            Button(
                onClick = { navController.navigate(route) },
                modifier = Modifier.fillMaxWidth().height(60.dp),
                shape = MaterialTheme.shapes.large
            ) {
                Text(label, fontSize = 18.sp)
            }
        }

        // Chatbot Button
        Button(
            onClick = { navController.navigate("chat") },
            modifier = Modifier.fillMaxWidth().height(60.dp),
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.tertiary),
            shape = MaterialTheme.shapes.large
        ) {
            Text("💬 AI Medical Chatbot", fontSize = 18.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun UniversalScanScreen(type: String, interpreter: Interpreter, historyList: MutableList<Report>, navController: NavController) {
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var resultText by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    val context = LocalContext.current

    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicturePreview()) {
        it?.let { bitmap = it; resultText = "" }
    }
    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            val stream = context.contentResolver.openInputStream(it)
            bitmap = BitmapFactory.decodeStream(stream)
            resultText = ""
        }
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("$type Analysis", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(20.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(onClick = { cameraLauncher.launch(null) }) { Text("📸 Camera") }
            Button(onClick = { galleryLauncher.launch("image/*") }) { Text("🖼️ Gallery") }
        }

        Spacer(modifier = Modifier.height(20.dp))

        bitmap?.let {
            Image(it.asImageBitmap(), null, modifier = Modifier.size(250.dp))
            Spacer(modifier = Modifier.height(20.dp))

            Button(onClick = {
                isLoading = true
                val score = runBinaryModel(it, interpreter)
                isLoading = false

                if (score == -1f) {
                    resultText = "Error: AI Processing failed."
                } else {
                    val isDetected = if (type == "CT Scan") score < 0.5f else score > 0.5f
                    val status = if (isDetected) "Detected 🔴" else "Normal 🟢"
                    resultText = "$type Analysis: $status\nConfidence: ${(score * 100).toInt()}%\n${getAdvice(score)}"
                    historyList.add(Report(type, resultText, System.currentTimeMillis().toString()))
                }
            }) {
                Text("Run AI Analysis")
            }
        }

        if (isLoading) CircularProgressIndicator(modifier = Modifier.padding(16.dp))

        if (resultText.isNotEmpty()) {
            Card(modifier = Modifier.padding(16.dp).fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(resultText, fontWeight = FontWeight.Medium)
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = { generatePDF(context, resultText) }) { Text("📄 Save") }
                        Button(onClick = { sharePDF(context, generatePDF(context, resultText)) }) { Text("📤 Share") }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))
        Button(onClick = { navController.popBackStack() }, colors = ButtonDefaults.buttonColors(MaterialTheme.colorScheme.error)) {
            Text("Cancel & Exit")
        }
    }
}

@Composable
fun BloodReportSection(navController: NavController) {
    var hb by remember { mutableStateOf("") }; var wbc by remember { mutableStateOf("") }; var platelets by remember { mutableStateOf("") }
    var result by remember { mutableStateOf("") }
    val context = LocalContext.current

    Column(modifier = Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Smart Blood Analysis", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(15.dp))
        OutlinedTextField(value = hb, onValueChange = { hb = it }, label = { Text("HB (Hemoglobin)") }, modifier = Modifier.fillMaxWidth())
        OutlinedTextField(value = wbc, onValueChange = { wbc = it }, label = { Text("WBC Count") }, modifier = Modifier.fillMaxWidth())
        OutlinedTextField(value = platelets, onValueChange = { platelets = it }, label = { Text("Platelets") }, modifier = Modifier.fillMaxWidth())

        Button(onClick = {
            val hVal = hb.toFloatOrNull() ?: 0f; val wVal = wbc.toFloatOrNull() ?: 0f; val pVal = platelets.toFloatOrNull() ?: 0f
            result = "Hemoglobin: ${checkHB(hVal)}\nWBC: ${checkWBC(wVal)}\nPlatelets: ${checkPlatelets(pVal)}"
        }, modifier = Modifier.padding(top = 20.dp).fillMaxWidth()) { Text("Analyze Blood Parameters") }

        if (result.isNotEmpty()) {
            Card(modifier = Modifier.padding(top = 20.dp).fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(result)
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = { generatePDF(context, result) }) { Text("📄 Save") }
                        Button(onClick = { sharePDF(context, generatePDF(context, result)) }) { Text("📤 Share") }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))
        Button(onClick = { navController.popBackStack() }) { Text("Back to Home") }
    }
}

@Composable
fun ChatScreen(navController: NavController) {
    var userInput by remember { mutableStateOf("") }
    var reply by remember { mutableStateOf("") }

    Column(modifier = Modifier.fillMaxSize().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("AI Doctor Chat 🩺", fontSize = 26.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(20.dp))
        OutlinedTextField(value = userInput, onValueChange = { userInput = it }, label = { Text("Ask symptoms...") }, modifier = Modifier.fillMaxWidth())
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = { reply = chatbotReply(userInput) }, modifier = Modifier.fillMaxWidth()) { Text("Ask AI Assistant") }

        if (reply.isNotEmpty()) {
            Card(modifier = Modifier.padding(top = 20.dp).fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
                Text(reply, modifier = Modifier.padding(16.dp))
            }
        }
        Spacer(modifier = Modifier.weight(1f))
        TextButton(onClick = { navController.popBackStack() }) { Text("Go Back") }
    }
}

// --- Logic & Core Functions ---

fun chatbotReply(input: String): String {
    val msg = input.lowercase()
    return when {
        msg.contains("fracture") -> "Possible bone fracture. Avoid weight-bearing and consult an orthopedist."
        msg.contains("mri") -> "MRI results require clinical correlation. Please show the report to a neurologist."
        msg.contains("pneumonia") -> "Lung infection signs detected. Consult a pulmonologist for antibiotics."
        msg.contains("serious") -> "Yes, medical conditions can escalate. Please seek professional help."
        else -> "I recommend consulting a doctor for a physical examination and better guidance."
    }
}

fun getAdvice(confidence: Float) = when {
    confidence > 0.85f -> "Urgent: High risk detected. Consult a doctor. ⚠️"
    confidence > 0.6f -> "Moderate: Clinical evaluation recommended."
    else -> "Results look normal. Keep monitoring. ✅"
}

fun checkHB(v: Float) = if (v < 13) "Low 🔴" else if (v > 17) "High 🔴" else "Normal 🟢"
fun checkWBC(v: Float) = if (v < 4000) "Low 🔴" else if (v > 11000) "High 🔴" else "Normal 🟢"
fun checkPlatelets(v: Float) = if (v < 150000) "Low 🔴" else if (v > 450000) "High 🔴" else "Normal 🟢"

fun generatePDF(context: Context, text: String): File {
    val pdf = PdfDocument()
    val page = pdf.startPage(PdfDocument.PageInfo.Builder(300, 600, 1).create())
    val canvas = page.canvas
    val paint = Paint().apply { textSize = 12f }
    canvas.drawText("MediScan AI - Medical Report", 50f, 50f, paint.apply { isFakeBoldText = true; textSize = 16f })
    val lines = text.split("\n")
    var yPos = 100f
    lines.forEach { line ->
        canvas.drawText(line.replace("🔴", "(Risk)").replace("🟢", "(Normal)"), 20f, yPos, paint.apply { isFakeBoldText = false; textSize = 12f })
        yPos += 25f
    }
    pdf.finishPage(page)
    val file = File(context.getExternalFilesDir(null), "Report_${System.currentTimeMillis()}.pdf")
    pdf.writeTo(FileOutputStream(file))
    pdf.close()
    return file
}

fun sharePDF(context: Context, file: File) {
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", file)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "application/pdf"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "Share Report"))
}

fun runBinaryModel(bitmap: Bitmap, interpreter: Interpreter): Float {
    return try {
        val inputShape = interpreter.getInputTensor(0).shape()
        val h = inputShape[1]; val w = inputShape[2]; val channels = inputShape[3]
        val resized = Bitmap.createScaledBitmap(bitmap, w, h, true)
        val byteBuffer = ByteBuffer.allocateDirect(4 * h * w * channels).apply { order(ByteOrder.nativeOrder()) }
        val intValues = IntArray(w * h)
        resized.getPixels(intValues, 0, w, 0, 0, w, h)
        for (pixel in intValues) {
            if (channels == 3) {
                byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
                byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
                byteBuffer.putFloat((pixel and 0xFF) / 255f)
            } else {
                byteBuffer.putFloat((((pixel shr 16) and 0xFF) + ((pixel shr 8) and 0xFF) + (pixel and 0xFF)) / 3f / 255f)
            }
        }
        val output = Array(1) { FloatArray(1) }
        interpreter.run(byteBuffer, output)
        output[0][0]
    } catch (e: Exception) { -1f }
}