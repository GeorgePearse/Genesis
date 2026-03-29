import pg from 'pg';

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
});

export interface User {
  id: number;
  clerk_user_id: string;
  email: string | null;
  created_at: Date;
}

export interface Subscription {
  id: number;
  user_id: number;
  stripe_customer_id: string | null;
  stripe_subscription_id: string | null;
  status: string;
  plan: string;
  current_period_end: Date | null;
  updated_at: Date;
}

export async function ensureUser(clerkUserId: string, email?: string): Promise<User> {
  const existing = await pool.query<User>(
    'SELECT * FROM users WHERE clerk_user_id = $1',
    [clerkUserId],
  );
  if (existing.rows.length > 0) {
    return existing.rows[0];
  }

  const inserted = await pool.query<User>(
    'INSERT INTO users (clerk_user_id, email) VALUES ($1, $2) RETURNING *',
    [clerkUserId, email ?? null],
  );
  return inserted.rows[0];
}

export async function getSubscription(userId: number): Promise<Subscription | null> {
  const result = await pool.query<Subscription>(
    'SELECT * FROM subscriptions WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 1',
    [userId],
  );
  return result.rows[0] ?? null;
}

export async function getSubscriptionByStripeCustomer(
  stripeCustomerId: string,
): Promise<Subscription | null> {
  const result = await pool.query<Subscription>(
    'SELECT * FROM subscriptions WHERE stripe_customer_id = $1',
    [stripeCustomerId],
  );
  return result.rows[0] ?? null;
}

export async function upsertSubscription(
  userId: number,
  data: {
    stripe_customer_id: string;
    stripe_subscription_id?: string;
    status: string;
    plan?: string;
    current_period_end?: Date;
  },
): Promise<Subscription> {
  const existing = await pool.query<Subscription>(
    'SELECT * FROM subscriptions WHERE user_id = $1',
    [userId],
  );

  if (existing.rows.length > 0) {
    const result = await pool.query<Subscription>(
      `UPDATE subscriptions
       SET stripe_customer_id = $1,
           stripe_subscription_id = COALESCE($2, stripe_subscription_id),
           status = $3,
           plan = COALESCE($4, plan),
           current_period_end = COALESCE($5, current_period_end),
           updated_at = now()
       WHERE user_id = $6
       RETURNING *`,
      [
        data.stripe_customer_id,
        data.stripe_subscription_id ?? null,
        data.status,
        data.plan ?? null,
        data.current_period_end ?? null,
        userId,
      ],
    );
    return result.rows[0];
  }

  const result = await pool.query<Subscription>(
    `INSERT INTO subscriptions
       (user_id, stripe_customer_id, stripe_subscription_id, status, plan, current_period_end)
     VALUES ($1, $2, $3, $4, $5, $6)
     RETURNING *`,
    [
      userId,
      data.stripe_customer_id,
      data.stripe_subscription_id ?? null,
      data.status,
      data.plan ?? 'free',
      data.current_period_end ?? null,
    ],
  );
  return result.rows[0];
}

export async function getUserByStripeCustomer(stripeCustomerId: string): Promise<User | null> {
  const sub = await getSubscriptionByStripeCustomer(stripeCustomerId);
  if (!sub) return null;
  const result = await pool.query<User>('SELECT * FROM users WHERE id = $1', [sub.user_id]);
  return result.rows[0] ?? null;
}

export { pool };
