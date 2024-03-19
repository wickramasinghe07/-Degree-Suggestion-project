// Import the functions you need from the SDKs you need
import { initializeApp, getApps } from "firebase/app";
import { getDatabase } from "firebase/database";


const firebaseConfig = {
    apiKey: "AIzaSyAcJPhbBGhQlSvOLkWC5ccpRIkpitlOEbc",
    authDomain: "uxdb-b0a22.firebaseapp.com",
    databaseURL: "https://uxdb-b0a22-default-rtdb.firebaseio.com",
    projectId: "uxdb-b0a22",
    storageBucket: "uxdb-b0a22.appspot.com",
    messagingSenderId: "459246476650",
    appId: "1:459246476650:web:1d355243c53a6f0af19913",
    measurementId: "G-H34SCZZRMC"
};


export const app = initializeApp(firebaseConfig);

export const database = getDatabase(app);