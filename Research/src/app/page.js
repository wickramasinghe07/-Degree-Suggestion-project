"use client"

import { useEffect, useRef, useState } from "react";
import { database } from "@/firebase/config";
import { onValue, ref, update, set, get, child } from 'firebase/database';

export default function testPage() {
    const elementRef = useRef(null);
    const sectionRef = useRef(null);
    const [show_Course_field, setshow_Course_field] = useState(false)
    const [courseName, setcourseName] = useState("")
    const [courseResData, setcourseResData] = useState("")
    const [courseValResData, setcourseValResData] = useState("")
    const [courseSkillResData, setcourseSkillResData] = useState("")
    const [show_Interest_field, setshow_Interest_field] = useState(false)
    const [Interest_fieldName, setInterest_fieldName] = useState("")
    const [InterestDegreeResData, setInterestDegreeResData] = useState("")

    useEffect(() => {
        const handleScroll = () => {
            if (sectionRef.current) {
                const rect = sectionRef.current.getBoundingClientRect();
                if (rect.top <= 300 && rect.top >= 0) {
                    console.log("🚀 ~ handleScroll ~ rect:", rect.top)
                    handelbutton();
                    showlog1();
                }
            }
        };

        window.addEventListener("scroll", handleScroll);

        return () => {
            window.removeEventListener("scroll", handleScroll);
        };
    }, []);

    useEffect(() => {

        const observer = new IntersectionObserver((entries) => {
            // console.log("🚀 ~ observer ~ entries:", entries)
            entries.forEach((entry) => {
                // console.log("🚀 ~ entries.forEach ~ entry:", entry)

                // if (entry.isIntersecting && entry.intersectionRect.top <= 500) {
                //     console.log("🚀 ~ entries.forEach ~ entry:", entry.intersectionRect.top)
                //     // The section is intersecting, call showlog
                //     showlog();
                // }
                if (entry.intersectionRatio > 0 && entry.boundingClientRect.top <= 500) {
                    // The section is intersecting and within 500 pixels from the top, call showlog
                    showlog();
                }
            });
        }, { threshold: 1 });

        // const observer = new IntersectionObserver((entries) => {
        //     entries.forEach((entry) => {
        //         if (entry.isIntersecting) {
        //             // The target element is now visible on the screen

        //             const rect = elementRef.current.getBoundingClientRect();
        //             if ( rect.y <= 500){
        //                 console.log("🚀🚀🚀 ~ entries.forEach ~ rect:", rect.y)
        //                 showlog();
        //             }
        //             console.log("🚀 ~ entries.forEach ~ rect:", rect.y)
        //             console.log("🚀 ~ useEffect ~ rect:", rect)
        //         }
        //     });
        // });

        if (elementRef.current) {
            observer.observe(elementRef.current);
        }

        // return () => {
        //     if (elementRef.current) {
        //         observer.unobserve(elementRef.current);
        //     }
        // };
    }, []);

    const showlog = () => {
        console.log("The specified section is showing on the screen.");
    };
    const showlog1 = () => {
        console.log("ggwp");
    };

    async function handeCourseButton() {
        setshow_Course_field(!show_Course_field)
    }

    async function handelbutton() {
        const db = database;

        const postData = {
            0: "sliit_IT",
            1: "sliit_DS",
            2: "sliit_SE",
        };

        try {
            set(ref(db, '/currentDegreeData'), postData)
                .then(() => {
                    console.log("Data set successfully");
                })
                .catch((error) => {
                    console.error("Error setting data:", error);
                });
        } catch (error) {
            console.log("error")
        }

        // const dbRef = ref(database);
        // get(child(dbRef, 'test')).then((snapshot) => {

        //     if (snapshot.exists()) {
        //         console.log(snapshot.val());
        //     } else {
        //         console.log("No data available");
        //     }
        // }).catch((error) => {
        //     console.error(error);
        // });

        // for main cam
        try {
            const response = await fetch('http://localhost:8002/show_cam');
            console.log("🚀 ~ handelbutton ~ response:", response)


        } catch (error) {
            console.error('Error fetching video stream:', error);
        }

    }

    const [result, setResult] = useState("");

    const handleSubmit = async (event) => {
        event.preventDefault();

        const AL_Stream = document.getElementById("AL_Stream").value;
        const Interest = document.getElementById("Interest").value;
        const courses = document.getElementById("courses").value;
        const cost = document.getElementById("cost").value;

        const data = {
            data: [AL_Stream, Interest, courses, cost]
        };

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const responseData = await response.json();
            const predictedLabel = responseData.predicted_label;
            setResult(predictedLabel)
            console.log("Predicted Label:", predictedLabel);

        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
        }
    };

    async function handelCourse() {
        console.log("courseName", courseName)
        const data = {
            courseName: courseName
        };

        try {
            const response = await fetch('http://localhost:8000/course', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const responseData = await response.json();
            setcourseResData(responseData.courseRes)
            console.log("responseData:", responseData);

        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
        }
    }


    async function handelAssistValue() {
        if (courseValResData != "") {
            setcourseValResData("")
        }
        const Interest = document.getElementById("Interest").value;
        console.log("Interest", Interest)
        const data = {
            InterestIn: Interest
        };
        try {
            const response = await fetch('http://localhost:8000/course_cost', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const responseData = await response.json();
            setcourseValResData(responseData.valueRes)
            console.log("responseData:", responseData);

        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
        }
    }

    async function handelAssistSkill() {

        const Interest = document.getElementById("Interest").value;
        const AL_Stream = document.getElementById("AL_Stream").value;

        console.log("Interest", Interest)
        const data = {
            InterestIn: Interest,
            AlStream: AL_Stream,
        };
        try {
            const response = await fetch('http://localhost:8000/skills', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const responseData = await response.json();
            setcourseSkillResData(responseData.valueRes)
            console.log("responseData:", responseData);

        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
        }
    }


    async function handelAssistInterest() {
        setshow_Interest_field(!show_Interest_field)
    }

    async function handelInterest_field() {
        console.log(Interest_fieldName)
        // InterestDegree
        const data = {
            InterestInDegree: Interest_fieldName
        };

        try {
            const response = await fetch('http://localhost:8000/InterestDegree', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const responseData = await response.json();
            setInterestDegreeResData(responseData.valueRes)
            console.log("responseData:", responseData);

        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
        }
    }

    return (

        <main>

            <header class="site-header" >
                <div class="section-overlay"></div>

                <div class="container">
                    <div class="row">

                        <div class="col-lg-12 col-12 text-center">
                            <h1 class="text-white">Degree Recommendation </h1>

                            <nav aria-label="breadcrumb">
                                <ol class="breadcrumb justify-content-center">
                                    <li class="breadcrumb-item"><a href="index.html">Home</a></li>

                                    <li class="breadcrumb-item active" aria-current="page">Degree Recommendation</li>
                                </ol>
                            </nav>
                        </div>

                    </div>
                </div>
            </header>

            <section class="section-padding pb-0 d-flex justify-content-center align-items-center" >
                <div class="container">
                    <div class="row">

                        <div class="col-lg-12 col-12">
                            <form class="custom-form hero-form" onSubmit={handleSubmit}>
                                <h3 class="text-white mb-3">Find your dream !</h3>

                                <div class="row">
                                    <div class="col-lg-6 col-md-6 col-12">
                                        <div className="pb-1 ">
                                            <button type="button" className="p-1 bg-white rounded-xl text-xs bg-transparent text-transparent">Need Assist</button>
                                        </div>
                                        <div class="input-group ">
                                            <span class="input-group-text" id="basic-addon1"><i class="bi-person custom-icon"></i></span>

                                            <input type="text" name="AL_Stream" id="AL_Stream" class="form-control" placeholder="AL Stream" required />
                                        </div>
                                    </div>
                                    <div class="col-lg-6 col-md-6 col-12">
                                        <div className="mb-1">
                                            <div className="pb-1 ">
                                                <button type="button" onClick={handelAssistInterest} className="p-1 bg-blue-400 rounded-xl text-xs text-white">Need Assist ?</button>
                                            </div>
                                            {
                                                show_Interest_field ?
                                                    <div className="border rounded border-white">
                                                        <input type="text" onChange={(event) => {
                                                            setInterest_fieldName(event.target.value)
                                                        }} name="Add_course_title " id="Add_course_title" class="form-control" placeholder="Enter Your Interested degree field" required />
                                                        <div className="pl-4 pb-1">
                                                            <div className="flex justify-center items-center">
                                                                <button type="button" onClick={handelInterest_field} className="bg-white rounded-lg p-1">check</button>
                                                            </div>
                                                            {
                                                                InterestDegreeResData != "" ?
                                                                    <span>{InterestDegreeResData}</span>
                                                                    : null
                                                            }
                                                        </div>
                                                    </div> : null
                                            }
                                        </div>

                                        <div class="input-group">
                                            <span class="input-group-text" id="basic-addon1"><i class="bi-balloon-heart custom-icon"></i></span>

                                            <input type="text" name="Interest" id="Interest" class="form-control" placeholder="Interest In" required />
                                        </div>
                                    </div>
                                    <div class="col-lg-6 col-md-6 col-12">
                                    <div className="mb-1">
                                        <div className="pb-1 ">
                                            <button type="button" onClick={handeCourseButton} className="p-1 bg-blue-400 rounded-xl text-xs text-white">Need Assist ?</button>
                                        </div>
                                        {
                                            show_Course_field ?
                                                <div className="border rounded border-white">
                                                    <input type="text" onChange={(event) => {
                                                        setcourseName(event.target.value)
                                                    }} name="Add_course_title " id="Add_course_title" class="form-control" placeholder="Enter Your Course title" required />
                                                    <div className="pl-4 pb-1">
                                                        <div  className="flex justify-center items-center">
                                                            <button type="button" onClick={handelCourse} className="bg-white rounded-lg p-1">check</button>
                                                        </div>
                                                        {
                                                            courseResData != "" ?
                                                                <span>{courseResData}</span>
                                                                : null
                                                        }

                                                    </div>
                                                </div> : null
                                        }
                                        </div>

                                        <div class="input-group">
                                            <span class="input-group-text" id="basic-addon1"><i class="bi-gem custom-icon"></i></span>

                                            <select class="form-select form-control" name="courses" id="courses" aria-label="Default select example">
                                                <option selected>Do you have any experience with IT-related courses?</option>
                                                <option value="0">Yes, I have taken several IT-related courses and find them beneficial</option>
                                                <option value="1">No, I haven't taken any IT-related courses yet, but I'm considering it.</option>
                                                <option value="2">Yes, but I didn't find them very useful for my career goals.</option>
                                                <option value="3">No, I don't believe IT-related courses are necessary for my field of interest.</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="col-lg-6 col-md-6 col-12">
                                        <div className="pb-1 ">
                                            <button type="button" onClick={handelAssistValue} className="p-1 bg-blue-400 rounded-xl text-xs text-white">Need Assist ?</button>
                                        </div>
                                        {
                                            courseValResData != "" ?
                                                <span>{courseValResData}</span> : null
                                        }
                                        <div class="input-group">
                                            <span class="input-group-text" id="basic-addon1"><i class="bi-cash  custom-icon"></i></span>

                                            <select class="form-select form-control" name="cost" id="cost" aria-label="Default select example">
                                                <option selected>Do you consider the cost and potential value when deciding on learning opportunities?</option>
                                                <option value="0">Yes, I carefully weigh the cost against the potential value and benefits.</option>
                                                <option value="1">No, I prioritize learning opportunities solely based on their relevance and content</option>
                                                <option value="2">Yes, but I tend to prioritize lower-cost options even if they may have less value.</option>
                                                <option value="3">No, I believe that investing in learning is essential regardless of the cost.</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="col-lg-4 col-md-4 col-12">
                                        <div class="input-group">
                                            <span class="input-group-text" id="basic-addon1"><i class="bi-clock  custom-icon"></i></span>
                                            <select class="form-select form-control" name="cost" id="cost" aria-label="Default select example">
                                                <option selected>Degree Type</option>
                                                <option value="0">part-time</option>
                                                <option value="1">full-time</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="col-lg-4 col-md-4 col-12">
                                        <div class="input-group">
                                            <span class="input-group-text" id="basic-addon1"><i class="bi-cash  custom-icon"></i></span>

                                            <select class="form-select form-control" name="cost" id="cost" aria-label="Default select example">
                                                <option selected>course fee</option>
                                                <option value="0">Less than 500,000</option>
                                                <option value="1">Between 500,000 and 1,000,000</option>
                                                <option value="2">Between 1,000,000 and 1,500,000</option>
                                                <option value="3">Greater than 2,000,000</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="col-lg-4 col-md-4 col-12">
                                        <div class="input-group">
                                            <span class="input-group-text" id="basic-addon1"><i class="bi-geo-alt-fill  custom-icon"></i></span>

                                            <select class="form-select form-control" name="cost" id="cost" aria-label="Default select example">
                                                <option selected>Location</option>
                                                <option value="0">Colombo</option>
                                                <option value="1">Kandy</option>
                                                <option value="2">Jaffna</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="col-lg-6 col-md-6 col-12">
                                        <div className="pb-1 ">
                                            <button type="button" onClick={handelAssistSkill} className="p-1 bg-blue-400 rounded-xl text-xs text-white">Need Assist ?</button>
                                        </div>
                                        {
                                            courseSkillResData != "" ?
                                                <span>{courseSkillResData}</span> : null
                                        }
                                        <div class="input-group">
                                            <span class="input-group-text" id="basic-addon1"><i class="bi-award custom-icon"></i></span>

                                            <input type="text" name="Skills" id="Skills" class="form-control" placeholder="Your Skills" required />
                                        </div>
                                    </div>


                                    <div class="col-lg-12 col-12">
                                        <button type="submit" class="form-control">
                                            Find Best Degree Programes for ME
                                        </button>
                                    </div>

                                    <div class="col-12">

                                        {
                                            result != "" &&
                                            <div class="d-flex flex-wrap align-items-center mt-4 mt-lg-0">
                                                <span class="text-white mb-lg-0 mb-md-0 me-2"> We suggest you:</span>
                                                <div>
                                                    <a href="job-listings.html" class="badge">{result}  Degree Programes</a>
                                                </div>
                                            </div>
                                        }

                                    </div>
                                </div>
                            </form>
                        </div>

                        <div class="col-lg-6 col-12">
                            <img src="images/4557388.png" class="hero-image img-fluid" alt="" />
                        </div>

                    </div>
                </div>
            </section>


            <section class="job-section section-padding" >
                {/* <button onClick={handelbutton} className="bg-red-500">click test</button> */}
                <div class="container">
                    <div class="row align-items-center">

                        <div class="col-lg-4 col-md-6 col-12" ref={sectionRef} >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">IT</a>
                                    </h4>

                                    <div class="d-flex align-items-center">

                                        <p className="font-semibold">BSc (Hons) in Information Technology Specialising in Information Technology </p>


                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            SLIIT
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 330,000 per semester
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="https://www.sliit.lk/computing/programmes/information-technology-degree/" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>

                        <div class="col-lg-4 col-md-6 col-12"  >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">DS</a>
                                    </h4>

                                    <div class="d-flex align-items-center">
                                        <p className="font-semibold">BSc (Hons) in Information Technology Specialising in Data Science </p>
                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            SLIIT
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 330,000(till year 2 Sem: 1), 350,000 (from year 2 sem: 2)
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="https://www.sliit.lk/computing/programmes/data-science-degree/" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>

                        <div class="col-lg-4 col-md-6 col-12"  >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">SE</a>
                                    </h4>

                                    <div class="d-flex align-items-center">

                                        <p className="font-semibold">BSc (Hons) in Information Technology Specialising in Software Engineering</p>

                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            SLIIT
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 330,000(till year 2 Sem: 1), 350,000 (from year 2 sem: 2)
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="https://www.sliit.lk/computing/programmes/software-engineering-degree/" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>
                        {/* add ref={sectionRef} here */}
                        <div class="col-lg-4 col-md-6 col-12"  >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">CS</a>
                                    </h4>

                                    <div class="d-flex align-items-center">

                                        <p className="font-semibold">BSc (Hons) Computer Science – (Plymouth University – United Kingdom)</p>


                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            NSBM
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 430,000 per semester
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="#" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>

                        <div class="col-lg-4 col-md-6 col-12"  >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">DS</a>
                                    </h4>

                                    <div class="d-flex align-items-center">

                                        <p className="font-semibold">BSc (Honours) in Data Science – (UGC Approved – Offered By NSBM) </p>


                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            NSBM
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 450,000 per semester
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="#" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>

                        <div class="col-lg-4 col-md-6 col-12"  >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">CN</a>
                                    </h4>

                                    <div class="d-flex align-items-center">

                                        <p className="font-semibold">BSc (Hons) Computer Networks – (Plymouth University – United Kingdom) </p>


                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            NSBM
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 330,000 per semester
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="#" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>

                        <div class="col-lg-4 col-md-6 col-12"  >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">SE</a>
                                    </h4>

                                    <div class="d-flex align-items-center">

                                        <p className="font-semibold">BSc (Hons) Software Engineering – (Plymouth University – United Kingdom)</p>


                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            NSBM
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 330,000 per semester
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="#" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>

                        <div class="col-lg-4 col-md-6 col-12"  >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">CS</a>
                                    </h4>

                                    <div class="d-flex align-items-center">

                                        <p className="font-semibold">Bachelor of Cyber Security - Deakin University, Australia</p>

                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            SLTC
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 260,000 per semester
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="#" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>

                        <div class="col-lg-4 col-md-6 col-12"  >
                            <div class="job-thumb job-thumb-box">

                                <div class="job-body" >
                                    <h4 class="job-title">
                                        <a href="job-details.html" class="job-title-link">SE</a>
                                    </h4>

                                    <div class="d-flex align-items-center">

                                        <p className="font-semibold">BSc (Hons) in Software Engineering</p>

                                    </div>

                                    <div class="d-flex align-items-center">
                                        <p class="job-location">
                                            <i class="custom-icon bi-geo-alt me-1"></i>
                                            SLTC
                                        </p>

                                        <p class="job-date">
                                            <i class="custom-icon bi-clock me-1"></i>
                                            4 Years
                                        </p>
                                    </div>

                                    <div class="d-flex align-items-center border-top pt-3">
                                        <p class="job-price mb-0">
                                            <i class="custom-icon bi-cash me-1"></i>
                                            Rs 330,000 per semester
                                        </p>

                                    </div>
                                    <div className="items-end justify-end flex">
                                        <a href="#" class="custom-btn btn ms-auto">Apply now</a>
                                    </div>

                                </div>
                            </div>
                        </div>

                        <div class="col-lg-12 col-12">
                            <nav aria-label="Page navigation example">
                                <ul class="pagination justify-content-center mt-5">
                                    <li class="page-item">
                                        <a class="page-link" href="#" aria-label="Previous">
                                            <span aria-hidden="true">Prev</span>
                                        </a>
                                    </li>

                                    <li class="page-item active" aria-current="page">
                                        <a class="page-link" href="#">1</a>
                                    </li>

                                    <li class="page-item">
                                        <a class="page-link" href="#">2</a>
                                    </li>

                                    <li class="page-item">
                                        <a class="page-link" href="#">3</a>
                                    </li>

                                    <li class="page-item">
                                        <a class="page-link" href="#">4</a>
                                    </li>

                                    <li class="page-item">
                                        <a class="page-link" href="#">5</a>
                                    </li>

                                    <li class="page-item">
                                        <a class="page-link" href="#" aria-label="Next">
                                            <span aria-hidden="true">Next</span>
                                        </a>
                                    </li>
                                </ul>
                            </nav>
                        </div>

                    </div>
                </div>
            </section>

            <section class="cta-section">
                <div class="section-overlay"></div>

                <div class="container">
                    <div class="row">

                    </div>
                </div>
            </section>
        </main>
    );
};