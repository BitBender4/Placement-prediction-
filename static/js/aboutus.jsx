function AboutUsPage() {
  const styles = {
    section: {
      backgroundColor: '#e7eaf6ff',
      padding: '50px 20px',
      fontFamily: 'Arial, sans-serif',
    },
    container: {
      maxWidth: '900px',
      margin: 'auto',
      textAlign: 'center',
    },
    title: {
      fontSize: '32px',
      color: '#333',
      marginBottom: '20px',
      marginTop: '1px',
    },
    text: {
      fontSize: '18px',
      color: '#555',
      lineHeight: '1.6',
      marginBottom: '20px',
    },
    subtitle: {
      fontSize: '24px',
      color: '#333',
      marginTop: '30px',
    },
    highlight: {
      fontWeight: 'bold',
    },
    iconRow: {
      marginTop: '30px',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      gap: '40px',
      flexWrap: 'wrap',
    },
    iconBox: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      textDecoration: 'none',
      color: '#333',
      fontSize: '16px',
    },
    iconImg: {
      width: '50px',
      height: '50px',
      marginBottom: '8px',
      cursor: 'pointer',
      transition: 'transform 0.2s',
    },
    label: {
      fontSize: '14px',
      fontWeight: '500',
      color: '#333',
      cursor: 'pointer',
      transition: 'color 0.2s, text-decoration 0.2s',
    },
  };

  return (
    <section style={styles.section}>
      <div style={styles.container}>
        <h2 style={styles.title}>About Us</h2>
        <p style={styles.text}>
          At <span style={styles.highlight}>Placement Prediction</span>, we are passionate about helping students and job seekers achieve their career goals with confidence.
          Our platform uses advanced analytics and AI-driven algorithms to <span style={styles.highlight}>predict placement opportunities</span> based on individual profiles,
          academic performance, skill sets, and industry trends.
        </p>
        <p style={styles.text}>
          We understand that landing the right job can be challenging, so our mission is to <span style={styles.highlight}>bridge the gap between education and employment </span>
          by providing accurate insights, personalized recommendations, and actionable steps to improve placement chances.
        </p>
        <p style={styles.text}>
          Whether you are a student preparing for campus recruitment or a professional looking to switch careers,
          <span style={styles.highlight}> Placement Prediction</span> is your trusted partner in guiding you towards the right opportunity.
        </p>
        <h3 style={styles.subtitle}>Our Vision</h3>
        <p style={styles.text}>
          To empower every learner and job seeker with data-backed career insights and ensure no talent goes unnoticed.
        </p>
        <h3 style={styles.subtitle}>Our Promise</h3>
        <p style={styles.text}>
          Transparency, accuracy, and innovation—helping you make smarter career decisions, every step of the way.
        </p>

        {/* Contact Section */}
        <h3 style={styles.subtitle}>Connect With Us</h3>
        <div style={styles.iconRow}>
          <a
            href="https://github.com/yourusername"
            target="_blank"
            style={styles.iconBox}
          >
            <img
              src="https://cdn-icons-png.flaticon.com/512/733/733553.png"
              alt="GitHub"
              style={styles.iconImg}
              onMouseOver={(e) => (e.currentTarget.style.transform = 'scale(1.2)')}
              onMouseOut={(e) => (e.currentTarget.style.transform = 'scale(1)')}
            />
            <span
              style={styles.label}
              onMouseOver={(e) => {
                e.currentTarget.style.color = '#0077b5';
                e.currentTarget.style.textDecoration = 'underline';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.color = '#333';
                e.currentTarget.style.textDecoration = 'none';
              }}
            >
              GitHub
            </span>
          </a>

          <a
            href="https://linkedin.com/in/yourusername"
            target="_blank"
            style={styles.iconBox}
          >
            <img
              src="https://cdn-icons-png.flaticon.com/512/174/174857.png"
              alt="LinkedIn"
              style={styles.iconImg}
              onMouseOver={(e) => (e.currentTarget.style.transform = 'scale(1.2)')}
              onMouseOut={(e) => (e.currentTarget.style.transform = 'scale(1)')}
            />
            <span
              style={styles.label}
              onMouseOver={(e) => {
                e.currentTarget.style.color = '#0077b5';
                e.currentTarget.style.textDecoration = 'underline';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.color = '#333';
                e.currentTarget.style.textDecoration = 'none';
              }}
            >
              LinkedIn
            </span>
          </a>

          <a
            href="mailto:yourmail@example.com"
            style={styles.iconBox}
          >
            <img
              src="https://cdn-icons-png.flaticon.com/512/732/732200.png"
              alt="Email"
              style={styles.iconImg}
              onMouseOver={(e) => (e.currentTarget.style.transform = 'scale(1.2)')}
              onMouseOut={(e) => (e.currentTarget.style.transform = 'scale(1)')}
            />
            <span
              style={styles.label}
              onMouseOver={(e) => {
                e.currentTarget.style.color = '#0077b5';
                e.currentTarget.style.textDecoration = 'underline';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.color = '#333';
                e.currentTarget.style.textDecoration = 'none';
              }}
            >
              Email
            </span>
          </a>
        </div>
      </div>
    </section>
  );
}

export default AboutUsPage;