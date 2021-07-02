
package org.tensorflow.lite.examples.posenet

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.snackbar.Snackbar
import kotlinx.android.synthetic.main.activity_main.*
import java.io.*
import java.util.*

class CameraActivity : AppCompatActivity() {

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)
    val objintent: Intent=intent
    val mySnackbar = Snackbar.make(linearLayout, "This app is an end-of-grade work. Created by Miguel Angel Cano Valero", 2000)

    //atributes of planks.kt
    var time = objintent.getStringExtra("time")
    var lh_le_ls_angle = objintent.getIntExtra("lh_le_ls_angle", 0)
    var rh_re_rs_angle = objintent.getIntExtra("rh_re_rs_angle", 0)
    var ls_lhip_la_angle = objintent.getIntExtra("ls_lhip_la_angle", 0)
    var rs_rhip_ra_angle = objintent.getIntExtra("rs_rhip_ra_angle", 0)

    //atributes of PushUp.kt
    var pushupreps = objintent.getIntExtra("pushupreps", 0)
    var plh_le_ls_angle = objintent.getIntExtra("plh_le_ls_angle", 0)
    var prh_re_rs_angle = objintent.getIntExtra("prh_re_rs_angle", 0)

    //atributes of PosenetActivity.kt
    var score = objintent.getIntExtra("reps",0)
    var lkneeangle = objintent.getIntExtra("lkneeangle", 0)
    var rkneeangle = objintent.getIntExtra("rkneeangle", 0)

    //atributes of Abs.kt
    var abs_reps = objintent.getIntExtra("abs_reps",0)
    var torso_angle= objintent.getIntExtra("torso_angle",0)



    //Create the spinner
    val spinner = findViewById<Spinner>(R.id.spinner)
    val lista = listOf("Plank", "Squats","Push-up","Abs")
    val adaptador = ArrayAdapter(this, android.R.layout.simple_spinner_item,lista)
    spinner.adapter=adaptador


    //create an if to comprobate the results chosing one type of exercise
    spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
      override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {

          if (spinner.selectedItem.equals("Squats")) {
            button.setOnClickListener {

              try {
                if (fileList().contains("notas.txt")) {
                  try {
                    val archivo = InputStreamReader(openFileInput("notas.txt"))
                    val br = BufferedReader(archivo)
                    var linea = br.readLine()
                    val todo = StringBuilder()
                    while (linea != null) {
                      todo.append(linea + "\n")
                      linea = br.readLine()
                    }
                    br.close()
                    archivo.close()
                    et1.setText(todo)
                  } catch (e: IOException) {
                  }
                }
                val archivo = OutputStreamWriter(openFileOutput("notas.txt", Activity.MODE_PRIVATE))
                archivo.write("Reps: " + score + "\n")
                archivo.write("Left knee angle: " + lkneeangle + "\n")
                archivo.write("Right knee angle: " + rkneeangle)
                archivo.flush()
                archivo.close()
              } catch (e: IOException) {
              }
            }

          } else if (spinner.selectedItem.equals("Plank")) {

            button.setOnClickListener {

              try {
                if (fileList().contains("notas.txt")) {
                  try {
                    val archivo = InputStreamReader(openFileInput("notas.txt"))
                    val br = BufferedReader(archivo)
                    var linea = br.readLine()
                    val todo = StringBuilder()
                    while (linea != null) {
                      todo.append(linea + "\n")
                      linea = br.readLine()
                    }
                    br.close()
                    archivo.close()
                    et1.setText(todo)
                  } catch (e: IOException) {
                  }
                }
                val archivo = OutputStreamWriter(openFileOutput("notas.txt", Activity.MODE_PRIVATE))
                archivo.write("Time: "  + time.toInt()/1000+ " s"+"\n")
                archivo.write("Left arm angle: " + lh_le_ls_angle + "\n")
                archivo.write("Right arm angle: " + rh_re_rs_angle+ "\n")
                archivo.write("Left body angle: " + ls_lhip_la_angle+ "\n")
                archivo.write("Right body angle: " + rs_rhip_ra_angle)
                archivo.flush()
                archivo.close()
              } catch (e: IOException) {
              }

            }

          } else if (spinner.selectedItem.equals("Push-up")) {
            button.setOnClickListener {
              try {
                if (fileList().contains("notas.txt")) {
                  try {
                    val archivo = InputStreamReader(openFileInput("notas.txt"))
                    val br = BufferedReader(archivo)
                    var linea = br.readLine()
                    val todo = StringBuilder()
                    while (linea != null) {
                      todo.append(linea + "\n")
                      linea = br.readLine()
                    }
                    br.close()
                    archivo.close()
                    et1.setText(todo)
                  } catch (e: IOException) {
                  }
                }
                val archivo = OutputStreamWriter(openFileOutput("notas.txt", Activity.MODE_PRIVATE))
                archivo.write("Reps: "  + pushupreps+ "\n")
                archivo.write("Left arm angle: " + plh_le_ls_angle + "\n")
                archivo.write("Right arm angle: " + prh_re_rs_angle)

                archivo.flush()
                archivo.close()
              } catch (e: IOException) {
              }

            }

          } else if (spinner.selectedItem.equals("Abs")) {
            button.setOnClickListener {
              try {
                if (fileList().contains("notas.txt")) {
                  try {
                    val archivo = InputStreamReader(openFileInput("notas.txt"))
                    val br = BufferedReader(archivo)
                    var linea = br.readLine()
                    val todo = StringBuilder()
                    while (linea != null) {
                      todo.append(linea + "\n")
                      linea = br.readLine()
                    }
                    br.close()
                    archivo.close()
                    et1.setText(todo)
                  } catch (e: IOException) {
                  }
                }
                val archivo = OutputStreamWriter(openFileOutput("notas.txt", Activity.MODE_PRIVATE))
                archivo.write("Reps: "  + abs_reps+ "\n")
                archivo.write("Torso angle: " + torso_angle)

                archivo.flush()
                archivo.close()
              } catch (e: IOException) {
              }

            }

          }
          else {
            Toast.makeText(this@CameraActivity, "Choose a workout to see the results", Toast.LENGTH_SHORT).show()
          }
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
        }

      }



    val botonir = findViewById<View>(R.id.bt_sent) as ImageButton
    botonir.setOnClickListener { // TODO Auto-generated method stub
      setContentView(R.layout.tfe_pn_activity_camera)
      savedInstanceState ?: supportFragmentManager.beginTransaction().replace(R.id.container, PosenetActivity()).commit()
    }


    val botonir3 = findViewById<View>(R.id.bt_plachas) as ImageButton
    botonir3.setOnClickListener { // TODO Auto-generated method stub
      setContentView(R.layout.tfe_pn_activity_camera)
      savedInstanceState ?: supportFragmentManager.beginTransaction()
        .replace(R.id.container, Planks())
        .commit()
    }

    val botonir4 = findViewById<View>(R.id.bt_abs) as ImageButton
    botonir4.setOnClickListener { // TODO Auto-generated method stub
      setContentView(R.layout.tfe_pn_activity_camera)
      savedInstanceState ?: supportFragmentManager.beginTransaction()
              .replace(R.id.container, Abs())
              .commit()
    }
    val botonir5 = findViewById<View>(R.id.bt_pushup) as ImageButton
    botonir5.setOnClickListener { // TODO Auto-generated method stub
      setContentView(R.layout.tfe_pn_activity_camera)
      savedInstanceState ?: supportFragmentManager.beginTransaction()
              .replace(R.id.container, PushUp())
              .commit()
    }

    //Button to send email
    val botonemail = findViewById<View>(R.id.bt_email) as Button
    botonemail.setOnClickListener{
      val email = "PUT YOUR EMAIL HERE"
      val intentEmail = Intent(Intent.ACTION_SEND, Uri.parse(email))
      intentEmail.type = "plain/text"
      intentEmail.putExtra(Intent.EXTRA_SUBJECT, "Results of the workout, send to yourself :D")
      intentEmail.putExtra(Intent.EXTRA_TEXT, "--YOUR ABS STATICTICS:"+ "\n" +"\t"+"abs reps: " + abs_reps+ "\n"+"\t" +"Torso angle: " + torso_angle+ "\n"+ "\n"+"A good angle for torso is an angle between 80 and 110" + "\n"
      + "**************************" + "\n" + "--YOUR SQUAT STATICTICS:"+ "\n" +"\t"+ "Reps: " + score + "\n"+"\t" +"Left knee angle: " + lkneeangle + "\n"+"\t"+"Right knee angle: " + rkneeangle+ "\n" + "\n"+ "A good angle for knees angle is an angle between 80 and 120"+ "\n"
              + "**************************" + "\n" +"--YOUR PUSH-UP STATICTICS:"+ "\n" +"\t"+ "Reps: "  + pushupreps+ "\n"+"\t" + "Left arm angle: " + plh_le_ls_angle + "\n"+"\t" + "Right arm angle: " + prh_re_rs_angle + "\n" + "\n"+ "A good angle for arms angle is an angle between 80 and 130"+ "\n"
                + "**************************" + "\n" +"--YOUR PLANK STATICTICS:"+ "\n"+"\t" + "Time: "  + time+ "\n"+"\t" + "Left arm angle: " + lh_le_ls_angle + "\n"+"\t" + "Left body angle: " + ls_lhip_la_angle+ "\n"+"\t" + "Right body angle: " + rs_rhip_ra_angle+ "\n"+  "\n"+"A good angle for arms angle is an angle between 60 and 100")

      intentEmail.putExtra(Intent.EXTRA_EMAIL, arrayOf("PUT YOUR EMAIL HERE"))
      startActivity(Intent.createChooser(intentEmail, "Elige cliente de correo"))
    }

    val btinfo =  findViewById<View>(R.id.bt_info) as ImageButton
    btinfo.setOnClickListener{
      mySnackbar.show()
    }

  }

}


